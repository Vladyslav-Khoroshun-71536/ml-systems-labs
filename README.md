# ML Systems — Laboratory Repository

Lab work for the **Machine Learning Systems** course.

## Repository layout

```
repo/
├── README.md
├── requirements.txt      # pinned versions (see "Deviations" below)
├── setup_env.bat         # one-click environment setup for Windows
├── .gitignore
├── data/                 # datasets, git-ignored, downloaded by scripts
├── lab01/
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/ … lab11/
└── project/
    ├── README.md
    ├── src/
    ├── results/
    ├── data_card.md
    └── model_card.md
```

## Environment

Python **3.11.8** in an isolated virtual environment.

**Windows (automatic):** run `setup_env.bat` from the repo root. It installs Python 3.11.8
per-user if missing, creates `.venv`, installs `requirements.txt` and writes
`lab01/results/versions.txt`.

**Manual:**

```
py -3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python lab01/src/print_versions.py > lab01/results/versions.txt
```

## Reproducibility rules

- Seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Timings: one warm-up, then at least 5 repeats, median reported. Single-sample inference: 100 repeats, median.
- Every lab records library versions in `results/versions.txt`.
- Reports contain results and explanations only; all code lives in `src/`.

## Deviations from the lab manual

| Package | Manual | Used | Reason |
|---|---|---|---|
| `pyarrow` | 16.1.0 | **15.0.2** | `mlflow==2.14.1` requires `pyarrow<16`; the manual's pins cannot be installed together (pip `ResolutionImpossible`). 15.0.2 is the newest compatible release. |

## Lab status

| Lab | Topic | Status |
|---|---|---|
| 01 | Environment and first system measurements | in progress |
