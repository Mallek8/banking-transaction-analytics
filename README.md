# Banking Transaction Analytics

[Français](README.fr.md)

Analysis of card transactions for **IFT599 / IFT799 — Data Science** at the Université de Sherbrooke (Fall 2025). The project turns a multi-file banking dataset into a single analytical table, then uses association rules, sequential patterns, and statistical comparison to describe everyday spending and to separate fraudulent transactions from legitimate ones.

The raw files are not in this repository. The notebook, the recommender, and the written report are.

**[Written report (PDF)](docs/IFT599-IFT799-TP1.pdf)** · **[Notebook](notebooks/transaction_analytics.ipynb)** · **[Recommender](src/recommender.py)**

## Highlights

| | |
| --- | --- |
| Source volume | 13,305,915 transactions, reduced to a reproducible 5% sample |
| Working table | 665,296 transactions, joined to cards, clients, MCC codes, and fraud labels |
| Fraud base rate | 672 labeled frauds (0.15%); mean amount **$104.50** versus **$42.60** for legitimate transactions |
| Association rules | Apriori, support ≥ 30%, confidence ≥ 50% → **3,894 rules**, mean lift **1.07** |
| Class separation | 15 of 24 tested variables differ at p < 0.05; Hotelling’s T² is significant. LDA accuracy is 99.7% and still misses most frauds |

Accuracy that high, on a 0.15% fraud rate, is mostly the model calling everything legitimate. The useful signal is elsewhere: amount, time of day, merchant category, and breaks in a client’s usual sequence.

## What the analysis covers

1. **Integrate** transactions, cards, users, merchant category codes, and fraud labels. Cards join on `card_id`, clients on `client_id`, merchants on `mcc`, labels as a binary `is_fraud`.
2. **Drop** fields that do not help the analysis: near-empty columns (`errors`, about 98% missing), duplicate client ids, and sensitive card attributes (`card_number`, `expires`, `cvv`).
3. **Clean** types (currency strings to floats, dates to datetimes), check duplicates, and impute merchant state and ZIP with the mode. Unlabeled `is_fraud` values stay missing.
4. **Engineer 21 features**: hour and time of day, weekend flag, amount quartiles and z-score, debt-to-income and credit utilization, age and credit-score bands, gap since the previous transaction, region, and merchant family (food, gas, retail, travel, healthcare, other).
5. **Describe merchants and time**: who concentrates volume, who concentrates revenue, and when transactions actually happen.
6. **Mine associations** with Apriori on categorical items (card, brand, merchant family, amount band, credit band, time of day).
7. **Compare classes** with chi-square, univariate tests, PCA, t-SNE, UMAP, Hotelling’s T², and linear discriminant analysis.
8. **Mine sequences** with a PrefixSpan-style pass over client histories, then score a small association-rule recommender.

Sampling uses `SAMPLE_SIZE = 0.05` and `RANDOM_STATE = 42`. Only the large transaction file is sampled. Cards, users, MCC codes, and fraud labels stay complete.

## Results worth keeping

**Merchants.** Three merchant ids (59935, 27092, 61195) account for about 42% of transaction counts. Revenue is more concentrated: merchant 27092 alone is about 35% of revenue in the top group. High count and high revenue are different businesses.

**Time.** Activity sits mainly between 7:00 and 16:00, with a peak near noon. Morning and afternoon together are more than three quarters of transactions. About 29% of transactions fall on the weekend. August is slightly busier; the rest of the year is flat.

**Card and merchant.** Card type and merchant family are associated (chi-square = 3078, p < 0.001).

| Card | Typical use | Mean amount |
| --- | --- | --- |
| Credit | Healthcare, travel, gas | $54.85 |
| Debit | Everyday food | $39.32 |
| Prepaid | Food and retail, almost no travel | $21.49 |

**Spending level by MCC.** Cruises, metals, hospitals, and airlines sit at the top of average ticket size (hundreds to about $1,500). Grocery, pharmacy, and personal care stay under $50. Food alone is on the order of 227,000 transactions in the sample, at a low average amount.

**Association rules.** The stable pattern is ordinary spending, not a rare combination. A representative rule:

`{merchant_food, card_debit} ⇒ {brand_mastercard, amount_low}`

Support 30.7%, confidence 58.6%, lift 1.19. Only four rules reach a lift of 1.2 or more. Mean support is 35.8%, mean confidence 67.8%, mean lift 1.072.

**Fraud profile.** Frauds are about 2.5 times larger on average, more variable (standard deviation about $193 versus $80), and more common in the afternoon (46% versus 37% of legitimate transactions). They are less common at night in this sample (3.6% versus 10.2%). Retail and travel are over-represented (risk ratios about 2.31 and 2.11). A simple rule that combines a high amount with a higher-risk merchant family covers about 16% of frauds while flagging about 4% of legitimate transactions. Amount plus late hour does not.

**Sequences.** On 100 clients, histories run from 144 to 2,085 transactions (mean 534). At 20% support, the frequent transitions are within the same merchant family (`Food → Food`, `Other → Other`) and a few daily loops (`Gas → Food`, `Other → Food`). Frauds in the sample tend to follow a break: a small daytime transaction, then a very large one in the evening or at night, sometimes within an hour, or a sudden jump in state or merchant family.

**Recommender.** `TransactionRecommender` ranks candidate items with a composite of confidence (0.4), lift (0.3), and support (0.3), and can explain which rule fired. On seven hand-built profiles and a set of 13 representative rules, 71% of profiles received at least one recommendation, and 60% of those recommendations met a confidence of 0.7.

**Geometry of the two classes.** PCA retains 97.2% of variance and shows frauds as outliers. t-SNE and UMAP do not isolate a fraud cluster; the points sit inside the legitimate cloud. That is why a linear classifier can look accurate and still fail as a detector.

## Stack

Python, pandas, NumPy, matplotlib, seaborn, SciPy, scikit-learn, mlxtend (Apriori), UMAP.

## Repository layout

```text
├── README.md
├── README.fr.md
├── requirements.txt
├── data/                  # local course files, not committed
├── docs/
│   └── IFT599-IFT799-TP1.pdf
├── notebooks/
│   └── transaction_analytics.ipynb
└── src/
    └── recommender.py    # TransactionRecommender
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the course files in `data/`:

- `transactions_data.csv`
- `cards_data.csv`
- `users_data.csv`
- `mcc_codes.json`
- `train_fraud_labels.json`

Then open `notebooks/transaction_analytics.ipynb`. The notebook resolves `data/` whether it is launched from the repository root or from `notebooks/`. The full 13 million rows are not required: the loader keeps 5% of the transaction file.

The recommender expects two frames produced in the notebook: association rules (`antecedents`, `consequents`, `support`, `confidence`, `lift`) and an item table with a `frequency` column.

```python
from src.recommender import TransactionRecommender

recommender = TransactionRecommender(rules_df, items_df)
result = recommender.recommend(
    ["merchant_food", "card_debit", "time_morning"],
    n_recommendations=5,
)
```

## Reading the metrics

- **Support** is how often the itemset appears. The threshold used here is 30%, so the rules describe common baskets, not rare events.
- **Confidence** is how often the consequent follows the antecedent. The threshold is 50%.
- **Lift** compares that confidence with what you would expect if the two sides were independent. Lift near 1.07 means a real but modest association.

## Limits

- The transaction file is a 5% sample. Rates in the report are rates inside that sample.
- About a third of transactions have no fraud label. Those rows were not imputed.
- Label encoding is used only for five low-cardinality fields. High-cardinality names (`merchant_city`, MCC descriptions) stay as text, because one-hot encoding would bury the association analysis.
- Global financial attributes (income, debt, credit limit) separate the classes less than amount, time, and merchant family.
- The recommender is a transparent baseline. Short histories fall back to popular items, and the module does not replace a sequence model.

## Authors

Course project, Université de Sherbrooke, Fall 2025. Instructor: Jordan Félicien Masakuna.

- Hannachi Mallek — [github.com/Mallek8](https://github.com/Mallek8)
- Mamady Iii Diakite
- Abdul Magide Djalo
