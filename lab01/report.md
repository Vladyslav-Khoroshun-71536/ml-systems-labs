# Lab 01 — Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment, train two baseline classifiers on the Breast Cancer Wisconsin
dataset, measure what each model costs as a system (training time, inference latency, model size, memory),
and decide on which deployment tier — Cloud, Edge, Mobile or TinyML — each model could run.

## 2. Method

- **Environment:** Python 3.11.8 in a `.venv`, all libraries pinned in `requirements.txt`; versions recorded
  in `results/versions.txt` by `src/print_versions.py`. One deviation from the manual: `pyarrow` 15.0.2
  instead of 16.1.0, because `mlflow==2.14.1` requires `pyarrow<16` (see the repository README).
- **Data:** `load_breast_cancer` (569 samples, 30 features), 70/30 stratified split, `random_state=42`
  → 398 train / 171 test. Seeds set to 42 for `random`, NumPy and torch.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and
  `RandomForestClassifier(n_estimators=100, random_state=42)`, default settings otherwise, no feature scaling.
- **Measurements** (`src/baselines.py`, run on a Windows laptop CPU):
  - training time: 1 warm-up, then median of 5 fits;
  - single-sample inference: 1 warm-up, then median of 100 `predict` calls on one test sample;
  - model size: `joblib.dump` to `results/model_logreg.joblib` and `results/model_rf.joblib`;
  - peak memory: process resident memory (RSS) sampled every 1 ms with `memory_profiler`
    during training and during 100 inferences.
- **Budget check:** a model fits a tier if its peak memory, latency and model size are all within the
  tier's upper limits from the manual.

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | **0.9415** |
| RandomForest | **0.9357** |

### System cost (`results/measurements.csv`)

| Model | Train time (median) | Inference, 1 sample (median) | Model size | Peak RSS train | Peak RSS inference | RSS increase while training |
|---|---|---|---|---|---|---|
| LogisticRegression | 272.7 ms | 0.042 ms | 1,055 B (1.03 KB) | 229.2 MB | 229.3 MB | 0.9 MB |
| RandomForest | 159.0 ms | 2.197 ms | 290,889 B (284.07 KB) | 230.4 MB | 230.4 MB | 2.1 MB |

Process memory before training was 228.4 MB. LogisticRegression did **not** converge within the required
1,000 iterations (lbfgs on unscaled features); the result is reported with the settings the manual requires.

### Deployment budget fit (`results/budget_fit.csv`)

| Tier | Budget (memory / latency / size) | LogisticRegression | RandomForest |
|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | ✅ fits | ✅ fits |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | ✅ fits | ✅ fits |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | ✅ fits (229 MB, close to the limit) | ✅ fits (230 MB, close to the limit) |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | ❌ memory | ❌ memory and size (284 KB) |

## 4. Conclusions

1. **On this dataset the simpler model is just as accurate and far cheaper to deploy.** Logistic regression
   reached 94.2% test accuracy against 93.6% for the random forest, while its saved model is about 275×
   smaller (1 KB vs 284 KB) and a single prediction is about 50× faster (0.04 ms vs 2.2 ms). The forest's
   extra cost buys nothing here, so model choice should look at system cost, not only at accuracy.

2. **The model is not the system — the runtime around it decides memory.** Training or predicting added
   only 1–2 MB, but the Python process itself already used about 228 MB (interpreter, NumPy, scikit-learn
   and torch, imported for seeding). That is why both models only barely fit the Mobile budget and neither
   fits TinyML: even the 1 KB logistic model cannot run on a 256 KB microcontroller inside a Python stack.
   It would need its 30 weights exported to plain C, and the forest would also have to shrink below 100 KB.

3. **Latency was never the bottleneck; memory and model size were.** Both models answer far below even
   the strictest 10 ms TinyML limit, so the tier choice was decided by memory and file size. The measurements
   also showed a trade-off that accuracy alone hides: the random forest trains faster (159 ms vs 273 ms,
   partly because logistic regression ran into its iteration limit on unscaled features) but predicts much
   slower, so the better choice depends on whether a system retrains often or serves many predictions.
