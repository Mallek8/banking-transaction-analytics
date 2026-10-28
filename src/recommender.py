
"""
Système de Recommandation basé sur les Règles d'Association
==========================================================

Ce module implémente un système de recommandation intelligent pour les transactions
bancaires basé sur l'analyse des règles d'association fréquentes.

Fonctionnalités:
- Recommandations basées sur les règles d'association Apriori
- Support pour différents types de recommandations (commerces, montants, périodes)
- Système de scoring et de ranking des recommandations
- Interface simple pour l'intégration dans des applications

Usage:
    from src.recommender import TransactionRecommender

    recommender = TransactionRecommender(rules_df, items_df)
    recommendations = recommender.recommend(client_history, n_recommendations=5)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TransactionRecommender:
    """
    Système de recommandation basé sur les règles d'association.
    
    Ce système utilise les règles d'association extraites par l'algorithme Apriori
    pour recommander des items complémentaires basés sur l'historique des transactions
    d'un client.
    """
    
    def __init__(self, rules_df: pd.DataFrame, items_df: pd.DataFrame):
        """
        Initialise le système de recommandation.
        
        Parameters:
        -----------
        rules_df : pd.DataFrame
            DataFrame contenant les règles d'association avec colonnes:
            - antecedents: frozenset des items antécédents
            - consequents: frozenset des items conséquents  
            - support: support de la règle
            - confidence: confiance de la règle
            - lift: lift de la règle
            
        items_df : pd.DataFrame
            DataFrame contenant les items disponibles avec leurs métadonnées
        """
        self.rules_df = rules_df.copy()
        self.items_df = items_df.copy()
        
        # Préparer les données pour la recommandation
        self._prepare_recommendation_data()
        
        # Statistiques du système
        self.n_rules = len(self.rules_df)
        self.n_items = len(self.items_df)
        
        print(f" Système de recommandation initialisé:")
        print(f"   • {self.n_rules:,} règles d'association chargées")
        print(f"   • {self.n_items:,} items disponibles")
        print(f"   • Prêt pour les recommandations")
    
    def _prepare_recommendation_data(self):
        """Prépare les données pour optimiser les recommandations."""
        
        # Convertir les frozensets en listes pour faciliter le traitement
        self.rules_df['antecedents_list'] = self.rules_df['antecedents'].apply(list)
        self.rules_df['consequents_list'] = self.rules_df['consequents'].apply(list)
        
        # Créer un index des règles par antécédent pour accélérer la recherche
        self.antecedent_index = {}
        for idx, row in self.rules_df.iterrows():
            for antecedent in row['antecedents_list']:
                if antecedent not in self.antecedent_index:
                    self.antecedent_index[antecedent] = []
                self.antecedent_index[antecedent].append(idx)
        
        # Calculer des scores composites pour le ranking
        self.rules_df['composite_score'] = (
            self.rules_df['confidence'] * 0.4 +
            self.rules_df['lift'] * 0.3 +
            self.rules_df['support'] * 0.3
        )
    
    def recommend(self, 
                  client_history: List[str], 
                  n_recommendations: int = 5,
                  min_confidence: float = 0.3,
                  min_lift: float = 1.0) -> Dict[str, any]:
        """
        Génère des recommandations basées sur l'historique du client.
        
        Parameters:
        -----------
        client_history : List[str]
            Liste des items de l'historique du client
            
        n_recommendations : int, default=5
            Nombre de recommandations à retourner
            
        min_confidence : float, default=0.3
            Confiance minimale des règles à considérer
            
        min_lift : float, default=1.0
            Lift minimal des règles à considérer
            
        Returns:
        --------
        Dict contenant:
            - recommendations: Liste des recommandations avec scores
            - matched_rules: Règles utilisées pour les recommandations
            - statistics: Statistiques de la recommandation
        """
        
        if not client_history:
            return self._get_empty_recommendation()
        
        # Trouver les règles applicables
        applicable_rules = self._find_applicable_rules(
            client_history, min_confidence, min_lift
        )
        
        if applicable_rules.empty:
            return self._get_fallback_recommendation(client_history)
        
        # Générer les recommandations
        recommendations = self._generate_recommendations(
            applicable_rules, client_history, n_recommendations
        )
        
        # Calculer les statistiques
        stats = self._calculate_recommendation_stats(
            client_history, applicable_rules, recommendations
        )
        
        return {
            'recommendations': recommendations,
            'matched_rules': applicable_rules.head(10).to_dict('records'),
            'statistics': stats
        }
    
    def _find_applicable_rules(self, 
                              client_history: List[str], 
                              min_confidence: float,
                              min_lift: float) -> pd.DataFrame:
        """Trouve les règles applicables basées sur l'historique du client."""
        
        applicable_rules = []
        
        for idx, rule in self.rules_df.iterrows():
            # Vérifier si les antécédents sont dans l'historique du client
            antecedents_in_history = all(
                ant in client_history for ant in rule['antecedents_list']
            )
            
            # Vérifier les seuils de qualité
            meets_thresholds = (
                rule['confidence'] >= min_confidence and
                rule['lift'] >= min_lift
            )
            
            if antecedents_in_history and meets_thresholds:
                # Calculer le score de pertinence
                relevance_score = self._calculate_relevance_score(
                    rule, client_history
                )
                
                rule_copy = rule.copy()
                rule_copy['relevance_score'] = relevance_score
                applicable_rules.append(rule_copy)
        
        if not applicable_rules:
            return pd.DataFrame()
        
        return pd.DataFrame(applicable_rules).sort_values(
            'relevance_score', ascending=False
        )
    
    def _calculate_relevance_score(self, 
                                  rule: pd.Series, 
                                  client_history: List[str]) -> float:
        """Calcule un score de pertinence pour une règle."""
        
        # Score de base basé sur les métriques de la règle
        base_score = rule['composite_score']
        
        # Bonus pour la correspondance exacte des antécédents
        exact_match_bonus = 1.0 if len(rule['antecedents_list']) == len(client_history) else 0.5
        
        # Bonus pour la diversité des conséquents
        diversity_bonus = min(1.0, len(rule['consequents_list']) / 3.0)
        
        # Score final
        relevance_score = base_score * (1 + exact_match_bonus) * (1 + diversity_bonus)
        
        return relevance_score
    
    def _generate_recommendations(self, 
                                 applicable_rules: pd.DataFrame,
                                 client_history: List[str],
                                 n_recommendations: int) -> List[Dict]:
        """Génère la liste finale des recommandations."""
        
        # Collecter tous les conséquents des règles applicables
        all_consequents = []
        for _, rule in applicable_rules.iterrows():
            for consequent in rule['consequents_list']:
                # Éviter de recommander ce que le client a déjà
                if consequent not in client_history:
                    all_consequents.append({
                        'item': consequent,
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'support': rule['support'],
                        'relevance_score': rule['relevance_score'],
                        'source_rule': f"{list(rule['antecedents'])} → {list(rule['consequents'])}"
                    })
        
        if not all_consequents:
            return []
        
        # Grouper par item et calculer le score agrégé
        item_scores = {}
        for rec in all_consequents:
            item = rec['item']
            if item not in item_scores:
                item_scores[item] = {
                    'item': item,
                    'max_confidence': 0,
                    'max_lift': 0,
                    'avg_support': 0,
                    'max_relevance': 0,
                    'rule_count': 0,
                    'source_rules': []
                }
            
            item_scores[item]['max_confidence'] = max(
                item_scores[item]['max_confidence'], rec['confidence']
            )
            item_scores[item]['max_lift'] = max(
                item_scores[item]['max_lift'], rec['lift']
            )
            item_scores[item]['max_relevance'] = max(
                item_scores[item]['max_relevance'], rec['relevance_score']
            )
            item_scores[item]['rule_count'] += 1
            item_scores[item]['source_rules'].append(rec['source_rule'])
        
        # Calculer les scores moyens
        for item in item_scores:
            item_scores[item]['avg_support'] = np.mean([
                rec['support'] for rec in all_consequents 
                if rec['item'] == item
            ])
        
        # Trier par score de pertinence et retourner le top N
        sorted_items = sorted(
            item_scores.values(),
            key=lambda x: x['max_relevance'],
            reverse=True
        )
        
        return sorted_items[:n_recommendations]
    
    def _calculate_recommendation_stats(self, 
                                       client_history: List[str],
                                       applicable_rules: pd.DataFrame,
                                       recommendations: List[Dict]) -> Dict:
        """Calcule les statistiques de la recommandation."""
        
        return {
            'client_history_size': len(client_history),
            'applicable_rules_count': len(applicable_rules),
            'recommendations_count': len(recommendations),
            'avg_rule_confidence': applicable_rules['confidence'].mean() if not applicable_rules.empty else 0,
            'avg_rule_lift': applicable_rules['lift'].mean() if not applicable_rules.empty else 0,
            'coverage': len(recommendations) / len(client_history) if client_history else 0
        }
    
    def _get_empty_recommendation(self) -> Dict:
        """Retourne une recommandation vide pour historique vide."""
        return {
            'recommendations': [],
            'matched_rules': [],
            'statistics': {
                'client_history_size': 0,
                'applicable_rules_count': 0,
                'recommendations_count': 0,
                'avg_rule_confidence': 0,
                'avg_rule_lift': 0,
                'coverage': 0
            }
        }
    
    def _get_fallback_recommendation(self, client_history: List[str]) -> Dict:
        """Retourne des recommandations de fallback quand aucune règle ne s'applique."""
        
        # Recommandations basées sur la popularité générale
        popular_items = self.items_df.nlargest(5, 'frequency')
        
        recommendations = []
        for _, item in popular_items.iterrows():
            if item['item'] not in client_history:
                recommendations.append({
                    'item': item['item'],
                    'confidence': 0.1,  # Faible confiance pour les fallbacks
                    'lift': 1.0,
                    'support': item['frequency'],
                    'relevance_score': 0.1,
                    'rule_count': 1,
                    'source_rules': ['Popularité générale']
                })
        
        return {
            'recommendations': recommendations[:5],
            'matched_rules': [],
            'statistics': {
                'client_history_size': len(client_history),
                'applicable_rules_count': 0,
                'recommendations_count': len(recommendations),
                'avg_rule_confidence': 0.1,
                'avg_rule_lift': 1.0,
                'coverage': len(recommendations) / len(client_history) if client_history else 0
            }
        }
    
    def get_item_info(self, item: str) -> Dict:
        """Retourne les informations détaillées sur un item."""
        
        item_info = self.items_df[self.items_df['item'] == item]
        if item_info.empty:
            return {'error': f'Item "{item}" non trouvé'}
        
        return item_info.iloc[0].to_dict()
    
    def explain_recommendation(self, item: str, client_history: List[str]) -> Dict:
        """Explique pourquoi un item est recommandé."""
        
        # Trouver les règles qui mènent à cette recommandation
        relevant_rules = self.rules_df[
            self.rules_df['consequents'].apply(lambda x: item in x)
        ]
        
        explanations = []
        for _, rule in relevant_rules.iterrows():
            if all(ant in client_history for ant in rule['antecedents_list']):
                explanations.append({
                    'rule': f"{list(rule['antecedents'])} → {list(rule['consequents'])}",
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
        
        return {
            'item': item,
            'explanations': explanations,
            'explanation_count': len(explanations)
        }
    
    def get_system_stats(self) -> Dict:
        """Retourne les statistiques générales du système."""
        
        return {
            'total_rules': self.n_rules,
            'total_items': self.n_items,
            'avg_confidence': self.rules_df['confidence'].mean(),
            'avg_lift': self.rules_df['lift'].mean(),
            'avg_support': self.rules_df['support'].mean(),
            'rules_by_confidence': {
                'high': len(self.rules_df[self.rules_df['confidence'] >= 0.7]),
                'medium': len(self.rules_df[(self.rules_df['confidence'] >= 0.5) & 
                                          (self.rules_df['confidence'] < 0.7)]),
                'low': len(self.rules_df[self.rules_df['confidence'] < 0.5])
            }
        }


def create_sample_recommendations():
    """
    Fonction utilitaire pour créer des exemples de recommandations.
    """
    
    # Exemples d'historiques clients
    sample_histories = [
        ['merchant_food', 'amount_medium', 'time_morning'],
        ['merchant_gas', 'amount_low', 'time_afternoon'],
        ['merchant_retail', 'merchant_food', 'amount_high'],
        ['time_evening', 'merchant_other', 'amount_very_high'],
        ['merchant_travel', 'time_night', 'amount_medium']
    ]
    
    return sample_histories


if __name__ == "__main__":
    print(" Système de Recommandation basé sur les Règles d'Association")
    print("=" * 60)
    print("Ce module doit être importé dans votre notebook principal.")
    print("Exemple d'utilisation :")
    print("  from src.recommender import TransactionRecommender")
    print("  recommender = TransactionRecommender(rules_df, items_df)")
    print("  recommendations = recommender.recommend(client_history)")