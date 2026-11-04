# Validation report

Rows checked: **48,842**, columns: **15**. Policy: violations are reported, **no rows are dropped** at this stage. `?` is the missing-value token and is an allowed category.

- Checks run: 47; FAIL: **0**; PASS: 30; INFO (missing values, duplicates): 17
- Rows with at least one missing value: **3,620** (7.41%)
- Exact duplicate rows: **52**; duplicate feature rows (ignoring label): **57**
- Label distribution: {'<=50K': 37155, '>50K': 11687}

## All checks

| check | column | violations | status | detail |
|---|---|---|---|---|
| columns present | all | 0 | PASS | missing=[] extra=[] |
| type int | age | 0 | PASS | stored dtype=int64 |
| type category | workclass | 0 | PASS | 8 distinct values |
| type int | fnlwgt | 0 | PASS | stored dtype=int64 |
| type category | education | 0 | PASS | 16 distinct values |
| type int | education-num | 0 | PASS | stored dtype=int64 |
| type category | marital-status | 0 | PASS | 7 distinct values |
| type category | occupation | 0 | PASS | 14 distinct values |
| type category | relationship | 0 | PASS | 6 distinct values |
| type category | race | 0 | PASS | 5 distinct values |
| type category | sex | 0 | PASS | 2 distinct values |
| type int | capital-gain | 0 | PASS | stored dtype=int64 |
| type int | capital-loss | 0 | PASS | stored dtype=int64 |
| type int | hours-per-week | 0 | PASS | stored dtype=int64 |
| type category | native-country | 0 | PASS | 41 distinct values |
| type category | income | 0 | PASS | 2 distinct values |
| range | age | 0 | PASS | allowed [17, 90], observed [17, 90] |
| range | hours-per-week | 0 | PASS | allowed [1, 99], observed [1, 99] |
| range | capital-gain | 0 | PASS | allowed [0, 99999], observed [0, 99999] |
| range | capital-loss | 0 | PASS | allowed [0, 4356], observed [0, 4356] |
| missing ('?') | age | 0 | INFO | 0.00% of rows |
| missing ('?') | workclass | 2799 | INFO | 5.73% of rows |
| missing ('?') | fnlwgt | 0 | INFO | 0.00% of rows |
| missing ('?') | education | 0 | INFO | 0.00% of rows |
| missing ('?') | education-num | 0 | INFO | 0.00% of rows |
| missing ('?') | marital-status | 0 | INFO | 0.00% of rows |
| missing ('?') | occupation | 2809 | INFO | 5.75% of rows |
| missing ('?') | relationship | 0 | INFO | 0.00% of rows |
| missing ('?') | race | 0 | INFO | 0.00% of rows |
| missing ('?') | sex | 0 | INFO | 0.00% of rows |
| missing ('?') | capital-gain | 0 | INFO | 0.00% of rows |
| missing ('?') | capital-loss | 0 | INFO | 0.00% of rows |
| missing ('?') | hours-per-week | 0 | INFO | 0.00% of rows |
| missing ('?') | native-country | 857 | INFO | 1.75% of rows |
| missing ('?') | income | 0 | INFO | 0.00% of rows |
| allowed categories | workclass | 0 | PASS | unknown=[] |
| allowed categories | education | 0 | PASS | unknown=[] |
| allowed categories | marital-status | 0 | PASS | unknown=[] |
| allowed categories | occupation | 0 | PASS | unknown=[] |
| allowed categories | relationship | 0 | PASS | unknown=[] |
| allowed categories | race | 0 | PASS | unknown=[] |
| allowed categories | sex | 0 | PASS | unknown=[] |
| allowed categories | native-country | 0 | PASS | unknown=[] |
| allowed categories | income | 0 | PASS | unknown=[] |
| education vs education-num | education | 0 | PASS | must match data dictionary |
| exact duplicate rows | all | 52 | INFO | removed later from TRAIN only |
| duplicate feature rows | 14 features | 57 | INFO | see label_quality.md |
