# Derived features: LogisticRegression with vs without

- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

Same split, same train-only cleaning; StandardScaler + OneHotEncoder; LogisticRegression(max_iter=2000, C=1.0). Test set: 14,653 rows.

| model | accuracy | F1 (>50K) |
|---|---|---|
| baseline (14 features) | 0.854 | 0.6601 |
| + capital_net, hours_per_age | 0.8578 | 0.6696 |

Change: accuracy +0.0038, F1 +0.0095.
