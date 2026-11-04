# Label-quality check

## 1. Duplicate-conflict audit

Rows whose 14 input features are identical form **54** duplicate groups. **5** of them have **conflicting income labels** (**10** rows). For these rows no model can be right on all copies, so they bound the achievable accuracy and indicate label noise (or missing features).

Examples (first conflicting groups):

| age | workclass | education | occupation | sex | hours-per-week | income |
|---|---|---|---|---|---|---|
| 34 | Private | Masters | Exec-managerial | Female | 50 | <=50K |
| 34 | Private | Masters | Exec-managerial | Female | 50 | >50K |
| 39 | Private | Bachelors | Craft-repair | Male | 40 | <=50K |
| 39 | Private | Bachelors | Craft-repair | Male | 40 | >50K |
| 41 | Private | HS-grad | Transport-moving | Male | 40 | <=50K |
| 41 | Private | HS-grad | Transport-moving | Male | 40 | >50K |
| 46 | Private | HS-grad | Craft-repair | Male | 40 | <=50K |
| 46 | Private | HS-grad | Craft-repair | Male | 40 | >50K |
| 46 | Private | HS-grad | Transport-moving | Male | 40 | <=50K |
| 46 | Private | HS-grad | Transport-moving | Male | 40 | >50K |

## 2. Manual re-check of 50 random rows

Sample: `results/label_audit_sample.csv` (seed 42). Each row was checked against the UCI data dictionary (allowed categories, value ranges, education ↔ education-num, relationship ↔ sex ↔ marital-status). Rows needing attention are marked `REVIEW`.

Flagged in the sample: **0 / 50**.

| rule | rows in full data | rows in 50-row sample |
|---|---|---|
| Husband but Female | 1 | 0 |
| Wife but Male | 3 | 0 |
| Husband/Wife but not married | 0 | 0 |
| gain and loss both > 0 | 0 | 0 |
| age < 20 with Masters/Doctorate/Prof-school | 1 | 0 |
| Never-worked but has occupation | 0 | 0 |
| age outside 17-90 | 0 | 0 |
| hours outside 1-99 | 0 | 0 |
| capital-gain = 99999 (top-code) | 244 | 0 |

No subjective labels are involved (income comes from the census), so the relabel-after-24-hours / Cohen's kappa protocol does not apply to Adult.
