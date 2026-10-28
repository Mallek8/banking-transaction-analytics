# Analytique des transactions bancaires

[English](README.md)

Analyse de transactions par carte pour le cours **IFT599 / IFT799 — Science des données**, Université de Sherbrooke, automne 2025. Le projet réunit plusieurs fichiers bancaires dans une seule table, puis décrit les dépenses courantes et les transactions frauduleuses à l’aide de règles d’association, de motifs séquentiels et de comparaisons statistiques.

Les fichiers sources ne sont pas dans ce dépôt. Le notebook, le recommandeur et le rapport écrit le sont.

**[Rapport (PDF)](docs/IFT599-IFT799-TP1.pdf)** · **[Notebook](notebooks/transaction_analytics.ipynb)** · **[Recommandeur](src/recommender.py)**

## En bref

| | |
| --- | --- |
| Volume source | 13 305 915 transactions, ramenées à un échantillon reproductible de 5 % |
| Table de travail | 665 296 transactions, jointes aux cartes, aux clients, aux codes MCC et aux étiquettes de fraude |
| Taux de fraude | 672 fraudes étiquetées (0,15 %) ; montant moyen **104,50 $** contre **42,60 $** pour les transactions légitimes |
| Règles d’association | Apriori, support ≥ 30 %, confiance ≥ 50 % → **3 894 règles**, lift moyen **1,07** |
| Séparation des classes | 15 variables sur 24 diffèrent à p < 0,05 ; le T² de Hotelling est significatif. La précision de l’ADL est de 99,7 % et laisse néanmoins passer la plupart des fraudes |

Une précision aussi élevée, avec un taux de fraude de 0,15 %, signifie surtout que le modèle classe presque tout comme légitime. Le signal utile est ailleurs : le montant, l’heure, la famille de commerçant, et les ruptures dans la séquence habituelle d’un client.

## Démarche

1. **Intégrer** les transactions, les cartes, les utilisateurs, les codes de catégorie commerçant et les étiquettes de fraude. Jointure des cartes sur `card_id`, des clients sur `client_id`, des commerçants sur `mcc`, et ajout de `is_fraud`.
2. **Retirer** ce qui n’éclaire pas l’analyse : la colonne presque vide `errors` (environ 98 % de manquants), les identifiants client dupliqués, et les attributs sensibles de carte (`card_number`, `expires`, `cvv`).
3. **Nettoyer** les types (montants en nombres, dates en datetime), vérifier les doublons, et imputer l’État et le code postal du commerçant par le mode. Les valeurs manquantes de `is_fraud` restent manquantes : ce sont des transactions non étiquetées.
4. **Construire 21 variables** : heure et période de la journée, indicateur de weekend, quartiles et score z du montant, ratio dette/revenu, utilisation du crédit, tranches d’âge et de score de crédit, délai depuis la transaction précédente, région, et famille de commerçant (alimentation, essence, détail, voyage, santé, autre).
5. **Décrire** les commerçants et le calendrier : qui concentre le volume, qui concentre le revenu, et à quels moments les transactions ont lieu.
6. **Extraire des associations** avec Apriori sur des items catégoriels (type de carte, marque, famille de commerçant, tranche de montant, tranche de crédit, période).
7. **Comparer les classes** par khi-deux, tests univariés, ACP, t-SNE, UMAP, T² de Hotelling et analyse discriminante linéaire.
8. **Extraire des séquences** par une passe de type PrefixSpan sur l’historique des clients, puis noter un recommandeur fondé sur les règles d’association.

L’échantillonnage utilise `SAMPLE_SIZE = 0.05` et `RANDOM_STATE = 42`. Seul le fichier de transactions est réduit. Cartes, utilisateurs, codes MCC et étiquettes de fraude sont conservés en entier.

## Résultats à retenir

**Commerçants.** Trois identifiants (59935, 27092, 61195) représentent environ 42 % du nombre de transactions. Le revenu est plus concentré : le commerçant 27092 pèse à lui seul environ 35 % du revenu du groupe de tête. Un fort volume et un fort revenu ne désignent pas les mêmes commerces.

**Temps.** L’activité se situe surtout entre 7 h et 16 h, avec un pic vers midi. La matinée et l’après-midi réunissent plus des trois quarts des transactions. Environ 29 % des transactions ont lieu le weekend. Août est un peu plus chargé ; le reste de l’année est stable.

**Carte et commerçant.** Le type de carte et la famille de commerçant sont associés (khi-deux = 3078, p < 0,001).

| Carte | Usage typique | Montant moyen |
| --- | --- | --- |
| Crédit | Santé, voyage, essence | 54,85 $ |
| Débit | Alimentation quotidienne | 39,32 $ |
| Prépayée | Alimentation et détail, presque pas de voyage | 21,49 $ |

**Niveau de dépense par MCC.** Croisières, métallurgie, hôpitaux et compagnies aériennes occupent le haut du ticket moyen (de quelques centaines de dollars à environ 1 500 $). Épicerie, pharmacie et soins personnels restent sous 50 $. L’alimentation représente à elle seule de l’ordre de 227 000 transactions dans l’échantillon, pour un montant moyen faible.

**Règles d’association.** Le motif stable est la dépense ordinaire, pas une combinaison rare. Règle représentative :

`{merchant_food, card_debit} ⇒ {brand_mastercard, amount_low}`

Support 30,7 %, confiance 58,6 %, lift 1,19. Seulement quatre règles atteignent un lift d’au moins 1,2. Support moyen 35,8 %, confiance moyenne 67,8 %, lift moyen 1,072.

**Profil de fraude.** Les fraudes sont en moyenne environ 2,5 fois plus élevées, plus variables (écart-type d’environ 193 $ contre 80 $), et plus fréquentes l’après-midi (46 % contre 37 % des transactions légitimes). Elles sont moins fréquentes la nuit dans cet échantillon (3,6 % contre 10,2 %). Le détail et le voyage sont surreprésentés (ratios de risque d’environ 2,31 et 2,11). Une règle simple qui combine un montant élevé et une famille de commerçants plus exposée couvre environ 16 % des fraudes tout en signalant environ 4 % des transactions légitimes. Montant élevé et heure tardive, pris ensemble, ne fonctionnent pas.

**Séquences.** Sur 100 clients, les historiques vont de 144 à 2 085 transactions (moyenne 534). Au seuil de support de 20 %, les transitions fréquentes restent dans la même famille (`Food → Food`, `Other → Other`) ou suivent une boucle quotidienne (`Gas → Food`, `Other → Food`). Les fraudes de l’échantillon tendent à suivre une rupture : une petite transaction en journée, puis un montant très élevé le soir ou la nuit, parfois en moins d’une heure, ou un changement brusque d’État ou de famille de commerçant.

**Recommandeur.** `TransactionRecommender` classe les items candidats avec un score composite de confiance (0,4), de lift (0,3) et de support (0,3), et peut expliquer quelle règle a été activée. Sur sept profils construits à la main et un jeu de 13 règles représentatives, 71 % des profils ont reçu au moins une recommandation, et 60 % de ces recommandations atteignent une confiance de 0,7.

**Géométrie des deux classes.** L’ACP conserve 97,2 % de la variance et montre les fraudes comme des points aberrants. Le t-SNE et l’UMAP n’isolent pas un groupe de fraudes : les points restent dans le nuage des transactions légitimes. C’est pour cela qu’un classifieur linéaire peut sembler précis et rester un mauvais détecteur.

## Environnement

Python, pandas, NumPy, matplotlib, seaborn, SciPy, scikit-learn, mlxtend (Apriori), UMAP.

## Organisation du dépôt

```text
├── README.md
├── README.fr.md
├── requirements.txt
├── data/                  # fichiers du cours, en local, non versionnés
├── docs/
│   └── IFT599-IFT799-TP1.pdf
├── notebooks/
│   └── transaction_analytics.ipynb
└── src/
    └── recommender.py    # TransactionRecommender
```

## Exécution

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Placer les fichiers du cours dans `data/` :

- `transactions_data.csv`
- `cards_data.csv`
- `users_data.csv`
- `mcc_codes.json`
- `train_fraud_labels.json`

Ouvrir ensuite `notebooks/transaction_analytics.ipynb`. Le notebook retrouve `data/` qu’il soit lancé depuis la racine du dépôt ou depuis `notebooks/`. Les 13 millions de lignes ne sont pas nécessaires : le chargeur conserve 5 % du fichier de transactions.

Le recommandeur attend deux tables produites dans le notebook : les règles d’association (`antecedents`, `consequents`, `support`, `confidence`, `lift`) et une table d’items avec une colonne `frequency`.

```python
from src.recommender import TransactionRecommender

recommender = TransactionRecommender(rules_df, items_df)
result = recommender.recommend(
    ["merchant_food", "card_debit", "time_morning"],
    n_recommendations=5,
)
```

## Lire les métriques

- Le **support** est la fréquence de l’ensemble d’items. Le seuil retenu est 30 % : les règles décrivent des paniers courants, pas des événements rares.
- La **confiance** est la fréquence du conséquent sachant l’antécédent. Le seuil est 50 %.
- Le **lift** compare cette confiance à celle attendue si les deux côtés étaient indépendants. Un lift proche de 1,07 est une association réelle, mais modeste.

## Limites

- Le fichier de transactions est un échantillon de 5 %. Les taux du rapport sont ceux de cet échantillon.
- Environ un tiers des transactions n’a pas d’étiquette de fraude. Ces lignes n’ont pas été imputées.
- L’encodage par étiquettes ne porte que sur cinq variables de faible cardinalité. Les noms à forte cardinalité (`merchant_city`, descriptions MCC) restent en texte : un encodage one-hot aurait noyé l’analyse d’association.
- Les attributs financiers globaux (revenu, dette, limite de crédit) séparent moins les classes que le montant, le temps et la famille de commerçant.
- Le recommandeur est une base de référence lisible. Les historiques courts retombent sur les items populaires, et le module ne remplace pas un modèle de séquences.

## Auteurs

Travail de session, Université de Sherbrooke, automne 2025. Enseignant : Jordan Félicien Masakuna.

- Hannachi Mallek — [github.com/Mallek8](https://github.com/Mallek8)
- Mamady Iii Diakite
- Abdul Magide Djalo
