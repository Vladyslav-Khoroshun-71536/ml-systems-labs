# Data card — UCI Adult

| Field | Value |
|---|---|
| Name | Adult / Census Income |
| Source | UCI Machine Learning Repository, donated 1996 by Ronny Kohavi and Barry Becker. Loaded via OpenML: `fetch_openml(name='adult', version=2)` (OpenML id 1590). |
| Original collection | Extract from the **1994 US Census Bureau** database (Current Population Survey). Rows were filtered: age > 16, hours > 0, fnlwgt > 1. |
| Licence | **CC BY 4.0** (UCI). Attribution: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 |
| Size | **48,842 rows × 15 columns** (14 features + label), ≈ 5 MB as CSV |
| Task | Binary classification: `income > 50K` |
| Label balance | `<=50K`: 37,155 (76.1 %); `>50K`: 11,687 (23.9 %) |
| Missing values | Coded as `?`. 3,620 rows (7.41 %) have at least one: `workclass` 2,799, `occupation` 2,809, `native-country` 857. |
| Duplicates | 52 exact duplicate rows; 5 groups (10 rows) with identical features but conflicting labels |
| Processed version | `data/processed/train.csv` (34,165 rows after removing 24 duplicates), `test.csv` (14,653 rows). SHA256 in `results/processed_data.sha256` and `results/run_config.yaml`. |

## Fields

The six numeric fields are `age` (17–90), `fnlwgt` (census sampling weight, i.e. how many people the row represents), `education-num` (1–16, a one-to-one coding of `education`), `capital-gain` (0–99,999; 99,999 is a top-code), `capital-loss` (0–4,356) and `hours-per-week` (1–99).

The eight categorical fields are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex` and `native-country`. The allowed values are listed in `config.yaml`.

## Known biases and limitations

- **Old data.** The data is from 1994. The $50K threshold is not inflation-adjusted (≈ $110K in 2026 money), and labour-market patterns have changed since. The data is not representative of today's population.
- **Sex imbalance and gap.** 66.8 % of the rows are men. 30.4 % of men are labelled `>50K`, against 10.9 % of women. A model will learn this historical pay gap and reproduce it.
- **Race imbalance and gap.** 85.5 % of the rows are White and 9.6 % Black, so the other groups have very few rows. `>50K` rates are 25.4 % for White and 12.1 % for Black people. Error rates will differ between groups.
- **Geography.** 91 % of the rows with a known country are from the US. Many countries have fewer than 30 rows.
- **Proxy features.** `relationship` (Husband/Wife) and `marital-status` encode sex and family status. Dropping `sex` alone does not remove the bias.
- **Missing values are not random.** `workclass` and `occupation` are missing together, mostly for people who are not working, and those people mostly earn `<=50K`. Imputing the mode hides this signal.
- **Label noise.** 5 groups of identical people carry different labels. There are also implausible rows, e.g. 1 "Husband" who is Female and 3 "Wife" who are Male; see `results/label_quality.md`.
- **Top-coding.** `capital-gain` is capped at 99,999 (244 rows), so true large values are unknown.

## Privacy considerations

- The data is anonymised: there are no names, addresses or IDs. It is public and widely used.
- It still contains **sensitive personal attributes**: race, sex, marital status, national origin and income. These are special-category data under GDPR Art. 9 (race) and protected characteristics in anti-discrimination law.
- Rare combinations of attributes can make re-identification possible in principle, e.g. a rare native country plus an unusual occupation plus an exact age. Do not join this data with other sources about individuals.
- Do not use models trained on this data for real decisions about people (credit, hiring, insurance) without a fairness audit.

## Intended use

Teaching and benchmarking of data pipelines and tabular classifiers. Not for real-world decisions.
