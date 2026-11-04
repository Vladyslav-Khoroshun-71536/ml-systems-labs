# Problem definition

**Task.** Predict whether a person's annual income exceeds **$50,000** from census attributes. This is a binary classification on the UCI Adult dataset (1994 US Census extract; 48,842 people).

**Inputs (14 features).**
- 6 numeric: `age`, `fnlwgt` (census sampling weight), `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- 8 categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.

**Output.** `income ∈ {<=50K, >50K}`. The positive class is `>50K`.

**Success metrics.** **F1 for the `>50K` class** (primary) and **accuracy** (secondary).
- The classes are imbalanced: only 23.9 % of people are `>50K`.
- A model that always predicts `<=50K` already gets 76 % accuracy but F1 = 0. This is why F1 is the main metric.
- The reference point is a LogisticRegression baseline on the 14 raw features.

**Constraints.**
1. **No leakage.** The 70/30 split is stratified and made *before* any cleaning. Every statistic is learned on the training part only: imputation values, clipping percentiles, duplicate removal, encoders and scaling. The test part is never cleaned, deduplicated or looked at during fitting.
2. **Reproducibility.**
    - One command (`python pipeline.py`) runs everything from the raw download.
    - All parameters live in `config.yaml`, and the seed is fixed (42).
    - The raw data is kept untouched.
    - The processed data is identified by a SHA256 hash, stored with the config in `results/run_config.yaml`.
3. **Transparency.**
    - Data problems (missing values, duplicates, range violations, conflicting labels) are reported, never silently dropped.
    - `data_card.md` documents the source, licence, biases and privacy risks.

**Out of scope.** Tuning the model for the best score. The model only measures whether the data and feature steps help. Fairness mitigation is also out of scope; the risks are documented in the data card.
