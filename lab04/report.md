# Lab 4 — Data pipeline and quality (UCI Adult)

Run everything with **one command**:

```bash
pip install -r requirements.txt
python pipeline.py
```

| File | Content |
|---|---|
| [`problem_definition.md`](problem_definition.md) | Task 1: inputs, output, metrics, constraints |
| [`pipeline.py`](pipeline.py) | Whole pipeline: ingest → validate → clean → features → label audit → hash |
| [`config.yaml`](config.yaml) | All parameters: schema, ranges, categories, split, seed |
| [`data_card.md`](data_card.md) | Task 7: source, licence, size, biases, privacy |
| `results/validation_report.md` | Task 3: all checks and violations |
| `results/preprocessor.joblib` | Task 4: fitted transformer (train only) |
| `results/feature_evaluation.md` | Task 5: LogisticRegression with vs without derived features |
| `results/label_quality.md`, `label_audit_sample.csv` | Task 6: duplicate-conflict audit and 50-row re-check |
| `results/run_config.yaml`, `processed_data.sha256` | Task 7: SHA256 of the processed data, stored with the config and library versions |
| `results/pipeline.log` | Log of every stage |

Data (`data/raw`, `data/processed`, `data/openml_cache`) is created by the pipeline and is git-ignored, because it is reproducible.

## 2. Ingestion

`fetch_openml(name='adult', version=2, as_frame=True)` → **48,842 rows × 15 columns**, logged in `pipeline.log`.

The raw frame is saved untouched to `data/raw/adult_raw.csv`, and its SHA256 is logged. All later stages work on a copy. The only standardisation is renaming OpenML's target `class` to `income` and representing `?` as missing.

## 3. Validation

47 checks are run on all 48,842 rows: columns present, types, ranges, allowed categories (from the UCI data dictionary), education ↔ education-num consistency, missing values and duplicates. **No row is dropped at this stage.**

| Result | Value |
|---|---|
| FAIL checks | **0**: types, ranges (age 17–90, hours 1–99, gain 0–99,999, loss 0–4,356), categories and education ↔ education-num all pass |
| Missing (`?`) | `workclass` 2,799 · `occupation` 2,809 · `native-country` 857 · **3,620 rows (7.41 %)** with at least one |
| Exact duplicates | **52** rows |
| Label balance | `<=50K` 37,155 / `>50K` 11,687 (23.9 %) |

The ranges in the manual equal the observed min/max exactly, so the values sit right at the boundaries: 90 is a top-code for age and 99,999 for capital-gain. The checks still matter. They would catch a broken download, a changed encoding or a new data version.

## 4. Cleaning and transformation (leakage-free)

The order is the point of this task:

1. **Split first**: 70/30, stratified by income, seed 42. That gives train 34,189 and test 14,653 rows, both with a positive rate of 0.239.
2. **Duplicates**: 24 exact duplicates are removed **from train only**, leaving 34,165 rows. The test set stays untouched, so it still represents real data.
3. **Imputer fitted on train**: mode for categorical features (`workclass` → Private, `occupation` → Prof-specialty, `native-country` → United-States), median for numeric ones. Numeric columns have no missing values.
4. **Clipping to the train 1st/99th percentiles**: age [17, 74], hours [8, 80], capital-gain [0, 15,024], capital-loss [0, 1,980], fnlwgt [27,035, 509,629].
5. The fitted transformer (clean → derived features → StandardScaler + OneHotEncoder) is saved with `joblib.dump` to `results/preprocessor.joblib`. At prediction time it is only **applied**, never refitted.

If the median, mode or percentiles were computed before the split, they would contain information from the test set. Test metrics would then be optimistic: this is data leakage.

The 99th-percentile clip squeezes capital-gain from 99,999 down to 15,024, so very large capital incomes become indistinguishable. This is a trade-off the manual prescribes; it makes the model robust but loses information.

## 5. Derived features

`capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. Since age ≥ 17, the denominator is never 0.

| Model (LogisticRegression) | Accuracy | F1 (>50K) |
|---|---:|---:|
| Baseline, 14 features | 0.8540 | 0.660 |
| + capital_net, hours_per_age | **0.8578** | **0.670** |

The gain is small (+0.4 pp accuracy, +0.95 pp F1). It is consistent with what the features add:
- `capital_net` merges gain and loss into one signed value. A linear model can then use a single weight for net capital income.
- `hours_per_age` adds a ratio, which a linear model cannot build from the raw columns itself.

Both models use the same split and train-only preprocessing, so the comparison is fair.

## 6. Label quality

- **Duplicate-conflict audit.** 54 groups of rows have identical 14 features, and **5 of these groups (10 rows) have conflicting labels**. Identical people cannot be both `<=50K` and `>50K`. Either the label is noisy or the features miss something that determines income, such as hours of overtime or region. These rows set an upper bound on accuracy.
- **50-row re-check.** 50 random rows (seed 42) are in `results/label_audit_sample.csv`. Each row was checked against the data dictionary: allowed values, ranges, education ↔ education-num, and relationship ↔ sex ↔ marital-status. **0 of 50** are impossible.
- **Full data.** The same rules applied to all rows find 1 "Husband" who is Female, 3 "Wife" who are Male, 1 person under 20 with a postgraduate degree, and 244 rows with top-coded capital-gain = 99,999.

Income is an objective census answer, not a subjective judgement. The relabel-after-24-hours / Cohen's kappa protocol is therefore not applicable to Adult.

## 7. Reproducibility

- `results/processed_data.sha256` holds the SHA256 of `train.csv`, `test.csv` and a combined hash. The same values are stored in `results/run_config.yaml`, next to the full config, the ingestion info and the library versions.
- Running the pipeline twice gives an identical hash. The CSVs are written with a fixed float format and LF line endings.
- The hash can differ with other pandas or scikit-learn versions; this is why the versions are recorded.
- The data card is in [`data_card.md`](data_card.md).
