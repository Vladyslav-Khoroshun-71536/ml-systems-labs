import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("lab4")

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
FEATURES = NUMERIC + CATEGORICAL
DERIVED = ["capital_net", "hours_per_age"]

def setup_logging(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(results_dir / "pipeline.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)

def ingest(cfg, raw_csv=None):
    d = cfg["data"]
    if raw_csv:
        cols = FEATURES_ORDER + ["class"]
        df = pd.read_csv(raw_csv, header=None, names=cols, skipinitialspace=True,
                         na_values=[cfg["schema"]["missing_token"]], keep_default_na=False)
        source = f"local file {raw_csv}"
    else:
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(name=d["openml_name"], version=d["openml_version"], as_frame=True,
                             data_home=str(ROOT / d["data_home"]))
        df = bunch.frame
        source = f"OpenML '{d['openml_name']}' v{d['openml_version']} (id {bunch.details.get('id')})"

    snap = ROOT / d["raw_snapshot"]
    snap.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(snap, index=False, na_rep=cfg["schema"]["missing_token"], lineterminator="\n")

    n_rows, n_cols = df.shape
    log.info(f"INGEST source: {source}")
    log.info(f"INGEST rows={n_rows} cols={n_cols} (expected {d['expected_rows']} x {d['expected_cols']})")
    log.info(f"INGEST raw snapshot: {snap.relative_to(ROOT)}  sha256={sha256_file(snap)[:16]}...")
    if (n_rows, n_cols) != (d["expected_rows"], d["expected_cols"]):
        log.warning("INGEST shape differs from the expected 48842 x 15!")
    return df, {"source": source, "rows": n_rows, "cols": n_cols, "snapshot_sha256": sha256_file(snap)}

FEATURES_ORDER = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                  "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                  "hours-per-week", "native-country"]


def standardize(df, cfg):
    out = df.copy()
    for alias in cfg["data"]["target_aliases"]:
        if alias in out.columns and alias != cfg["data"]["target"]:
            out = out.rename(columns={alias: cfg["data"]["target"]})
    tok = cfg["schema"]["missing_token"]
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            s = out[c].astype("object")
            s = s.where(s.isna(), s.astype(str).str.strip())
            out[c] = s.replace(tok, np.nan)
    t = cfg["data"]["target"]
    out[t] = out[t].str.rstrip(".")
    return out

def validate(raw, cfg, results_dir):
    sch = cfg["schema"]
    tok = sch["missing_token"]
    df = standardize(raw, cfg)
    rep = {"n_rows": len(df), "checks": []}

    def add(check, column, violations, detail=""):
        rep["checks"].append({"check": check, "column": column, "violations": int(violations),
                              "status": "PASS" if violations == 0 else "FAIL", "detail": detail})

    # schema: columns present
    expected = list(sch["types"])
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    add("columns present", "all", len(missing_cols) + len(extra_cols),
        f"missing={missing_cols} extra={extra_cols}")

    # types
    for col, typ in sch["types"].items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if typ == "int":
            num = pd.to_numeric(s, errors="coerce")
            bad = int(num.isna().sum() + ((num % 1) != 0).sum())
            add("type int", col, bad, f"stored dtype={raw[col].dtype if col in raw else df[col].dtype}")
        else:
            bad = int(s.map(lambda v: not isinstance(v, str)).sum())
            add("type category", col, bad, f"{s.nunique()} distinct values")

    # ranges
    for col, (lo, hi) in sch["ranges"].items():
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s[(s < lo) | (s > hi)]
        add("range", col, len(bad), f"allowed [{lo}, {hi}], observed [{s.min():g}, {s.max():g}]")

    miss = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        miss[col] = n
        rep["checks"].append({"check": "missing ('?')", "column": col, "violations": n,
                              "status": "INFO", "detail": f"{n / len(df):.2%} of rows"})
    rep["missing"] = miss
    rep["rows_with_any_missing"] = int(df.isna().any(axis=1).sum())

    for col, allowed in sch["categories"].items():
        s = df[col].dropna()
        unknown = sorted(set(s) - set(allowed))
        add("allowed categories", col, int(s.isin(unknown).sum()), f"unknown={unknown[:10]}")

    mp = sch["education_num"]
    exp = df["education"].map(mp)
    bad = df["education-num"].notna() & exp.notna() & (pd.to_numeric(df["education-num"]) != exp)
    add("education vs education-num", "education", int(bad.sum()), "must match data dictionary")

    dup_all = int(df.duplicated().sum())
    dup_feat = int(df.duplicated(subset=FEATURES).sum())
    rep["checks"].append({"check": "exact duplicate rows", "column": "all", "violations": dup_all,
                          "status": "INFO", "detail": "removed later from TRAIN only"})
    rep["checks"].append({"check": "duplicate feature rows", "column": "14 features",
                          "violations": dup_feat, "status": "INFO", "detail": "see label_quality.md"})
    rep["duplicates_exact"] = dup_all

    vc = df[cfg["data"]["target"]].value_counts(dropna=False)
    rep["label_distribution"] = {str(k): int(v) for k, v in vc.items()}

    (results_dir / "validation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    table = pd.DataFrame(rep["checks"])
    fails = table[table.status == "FAIL"]
    lines = [
        "# Validation report",
        "",
        f"Rows checked: **{len(df):,}**, columns: **{df.shape[1]}**. Policy: violations are reported, "
        "**no rows are dropped** at this stage. `?` is the missing-value token and is an allowed category.",
        "",
        f"- Checks run: {len(table)}; FAIL: **{len(fails)}**; PASS: {int((table.status == 'PASS').sum())}; "
        f"INFO (missing values, duplicates): {int((table.status == 'INFO').sum())}",
        f"- Rows with at least one missing value: **{rep['rows_with_any_missing']:,}** "
        f"({rep['rows_with_any_missing'] / len(df):.2%})",
        f"- Exact duplicate rows: **{dup_all}**; duplicate feature rows (ignoring label): **{dup_feat}**",
        f"- Label distribution: {rep['label_distribution']}",
        "",
        "## All checks",
        "",
        md_table(table),
        "",
    ]
    (results_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"VALIDATE {len(table)} checks, {len(fails)} FAIL, "
             f"{rep['rows_with_any_missing']} rows with missing values, {dup_all} exact duplicates")
    for _, r in fails.iterrows():
        log.warning(f"VALIDATE FAIL {r.check} [{r.column}]: {r.violations} ({r.detail})")
    return df, rep

class QuantileClipper(BaseEstimator, TransformerMixin):

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)


def add_derived(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def make_cleaner(cfg):
    c = cfg["cleaning"]
    lo, hi = c["clip_quantiles"]
    ct = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy=c["numeric_impute"])),
                          ("clip", QuantileClipper(lo, hi))]), NUMERIC),
        ("cat", SimpleImputer(strategy=c["categorical_impute"]), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def clean(df, cfg, results_dir, processed_dir):
    t = cfg["data"]["target"]
    X = df[FEATURES].copy()
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col]).astype(float)
    y = (df[t] == cfg["data"]["positive_label"]).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=cfg["seed"],
        stratify=y if cfg["split"]["stratify"] else None)
    log.info(f"CLEAN split: train={len(X_tr)} test={len(X_te)} "
             f"(positive rate {y_tr.mean():.3f} / {y_te.mean():.3f})")

    n0 = len(X_tr)
    if cfg["cleaning"]["drop_duplicates_in_train"]:
        keep = ~pd.concat([X_tr, y_tr.rename(t)], axis=1).duplicated()
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    n_dup = n0 - len(X_tr)
    log.info(f"CLEAN dropped {n_dup} exact duplicates from train; test untouched ({len(X_te)} rows)")

    cleaner = make_cleaner(cfg).fit(X_tr)
    tr = add_derived(cleaner.transform(X_tr))
    te = add_derived(cleaner.transform(X_te))
    tr[t], te[t] = y_tr.values, y_te.values

    num = cleaner.named_transformers_["num"]
    stats = pd.DataFrame({
        "feature": NUMERIC,
        "train median (imputer)": num.named_steps["impute"].statistics_,
        "clip low (p1)": num.named_steps["clip"].low_,
        "clip high (p99)": num.named_steps["clip"].high_,
    })
    modes = dict(zip(CATEGORICAL, cleaner.named_transformers_["cat"].statistics_))

    encoder = ColumnTransformer([("num", StandardScaler(), NUMERIC + DERIVED),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    from sklearn.preprocessing import FunctionTransformer
    preprocessor = Pipeline([("clean", cleaner),
                             ("derive", FunctionTransformer(add_derived)),
                             ("encode", encoder)])
    preprocessor.fit(X_tr)
    joblib.dump(preprocessor, results_dir / "preprocessor.joblib")
    log.info("CLEAN saved fitted transformer -> results/preprocessor.joblib")

    write_csv(tr, processed_dir / "train.csv")
    write_csv(te, processed_dir / "test.csv")
    info = {"train_rows": len(tr), "test_rows": len(te), "train_duplicates_dropped": n_dup,
            "imputer_modes": modes, "numeric_stats": stats.round(3).to_dict(orient="records")}
    return (X_tr, y_tr, X_te, y_te), info, stats, modes

def evaluate_features(splits, cfg, results_dir):
    X_tr, y_tr, X_te, y_te = splits
    m = cfg["model"]["logistic_regression"]
    rows = []
    for name, extra in [("baseline (14 features)", []), ("+ capital_net, hours_per_age", DERIVED)]:
        enc = ColumnTransformer([("num", StandardScaler(), NUMERIC + extra),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
        steps = [("clean", make_cleaner(cfg))]
        if extra:
            from sklearn.preprocessing import FunctionTransformer
            steps.append(("derive", FunctionTransformer(add_derived)))
        model = Pipeline(steps + [("encode", enc),
                                  ("lr", LogisticRegression(max_iter=m["max_iter"], C=m["C"]))])
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        rows.append({"model": name, "accuracy": round(accuracy_score(y_te, p), 4),
                     "F1 (>50K)": round(f1_score(y_te, p), 4)})
        log.info(f"FEATURES {name}: accuracy={rows[-1]['accuracy']} F1={rows[-1]['F1 (>50K)']}")
    res = pd.DataFrame(rows)
    d_acc = res.accuracy[1] - res.accuracy[0]
    d_f1 = res["F1 (>50K)"][1] - res["F1 (>50K)"][0]
    text = ["# Derived features: LogisticRegression with vs without", "",
            "- `capital_net = capital-gain - capital-loss`",
            "- `hours_per_age = hours-per-week / (age - 16)`", "",
            "Same split, same train-only cleaning; StandardScaler + OneHotEncoder; "
            f"LogisticRegression(max_iter={m['max_iter']}, C={m['C']}). Test set: {len(y_te):,} rows.", "",
            md_table(res), "",
            f"Change: accuracy {d_acc:+.4f}, F1 {d_f1:+.4f}.", ""]
    (results_dir / "feature_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    return res

def plausibility_flags(df):
    flags = pd.DataFrame(index=df.index)
    rel, sex, mar = df["relationship"], df["sex"], df["marital-status"]
    flags["Husband but Female"] = (rel == "Husband") & (sex == "Female")
    flags["Wife but Male"] = (rel == "Wife") & (sex == "Male")
    flags["Husband/Wife but not married"] = rel.isin(["Husband", "Wife"]) & \
        ~mar.isin(["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"])
    flags["gain and loss both > 0"] = (df["capital-gain"] > 0) & (df["capital-loss"] > 0)
    flags["age < 20 with Masters/Doctorate/Prof-school"] = (df["age"] < 20) & \
        df["education"].isin(["Masters", "Doctorate", "Prof-school"])
    flags["Never-worked but has occupation"] = (df["workclass"] == "Never-worked") & \
        df["occupation"].notna()
    flags["age outside 17-90"] = (df["age"] < 17) | (df["age"] > 90)
    flags["hours outside 1-99"] = (df["hours-per-week"] < 1) | (df["hours-per-week"] > 99)
    flags["capital-gain = 99999 (top-code)"] = df["capital-gain"] == 99999
    return flags


def label_quality(df, cfg, results_dir):
    t = cfg["data"]["target"]
    key = df[FEATURES].astype(str)
    g = pd.concat([key, df[t]], axis=1).groupby(FEATURES, dropna=False)[t]
    n_labels = g.nunique()
    sizes = g.size()
    conflict_groups = n_labels[n_labels > 1]
    rows_in_conflict = int(sizes[conflict_groups.index].sum())
    dup_groups = int((sizes > 1).sum())
    log.info(f"LABELS duplicate feature groups={dup_groups}, with conflicting labels={len(conflict_groups)} "
             f"({rows_in_conflict} rows)")
    examples = pd.DataFrame()
    if len(conflict_groups):
        ex_keys = conflict_groups.index[:5]
        idx = key.set_index(FEATURES).index.isin(ex_keys)
        examples = df.loc[idx, ["age", "workclass", "education", "occupation", "sex",
                                "hours-per-week", t]].sort_values(["age", "occupation", t])

    sample = df.sample(cfg["label_quality"]["sample_size"], random_state=cfg["seed"])
    flags = plausibility_flags(sample)
    sample_out = sample.copy()
    sample_out["flags"] = flags.apply(lambda r: "; ".join(r.index[r]), axis=1)
    sample_out.insert(0, "row_id", sample.index)
    sample_out["manual_check"] = np.where(sample_out["flags"] == "", "OK", "REVIEW")
    write_csv(sample_out, results_dir / "label_audit_sample.csv")

    full_flags = plausibility_flags(df).sum().rename("rows in full data").to_frame()
    full_flags["rows in 50-row sample"] = flags.sum()
    full_flags = full_flags.reset_index().rename(columns={"index": "rule"})
    n_flagged = int((sample_out["flags"] != "").sum())

    text = ["# Label-quality check", "",
            "## 1. Duplicate-conflict audit", "",
            f"Rows whose 14 input features are identical form **{dup_groups}** duplicate groups. "
            f"**{len(conflict_groups)}** of them have **conflicting income labels** "
            f"(**{rows_in_conflict}** rows). For these rows no model can be right on all copies, "
            "so they bound the achievable accuracy and indicate label noise (or missing features).", ""]
    if len(examples):
        text += ["Examples (first conflicting groups):", "", md_table(examples), ""]
    text += ["## 2. Manual re-check of 50 random rows", "",
             f"Sample: `results/label_audit_sample.csv` (seed {cfg['seed']}). Each row was checked against the "
             "UCI data dictionary (allowed categories, value ranges, education ↔ education-num, "
             "relationship ↔ sex ↔ marital-status). Rows needing attention are marked `REVIEW`.", "",
             f"Flagged in the sample: **{n_flagged} / {len(sample)}**.", "",
             md_table(full_flags), "",
             "No subjective labels are involved (income comes from the census), so the "
             "relabel-after-24-hours / Cohen's kappa protocol does not apply to Adult.", ""]
    (results_dir / "label_quality.md").write_text("\n".join(text), encoding="utf-8")
    return {"duplicate_groups": dup_groups, "conflicting_groups": int(len(conflict_groups)),
            "rows_in_conflict": rows_in_conflict, "sample_flagged": n_flagged,
            "rule_counts_full": dict(zip(full_flags.rule, full_flags["rows in full data"].astype(int)))}

def main():
    ap = argparse.ArgumentParser(description="Lab 4 data pipeline (one command).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-csv", default=None,
                    help="offline fallback: UCI-format adult CSV instead of fetch_openml")
    args = ap.parse_args()

    cfg_path = (ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    results_dir = ROOT / cfg["output"]["results_dir"]
    processed_dir = ROOT / cfg["output"]["processed_dir"]
    setup_logging(results_dir)
    np.random.seed(cfg["seed"])

    raw, ingest_info = ingest(cfg, args.raw_csv)
    df, val = validate(raw, cfg, results_dir)
    splits, clean_info, stats, modes = clean(df, cfg, results_dir, processed_dir)
    feat = evaluate_features(splits, cfg, results_dir)
    labels = label_quality(df, cfg, results_dir)

    h_tr = sha256_file(processed_dir / "train.csv")
    h_te = sha256_file(processed_dir / "test.csv")
    combined = hashlib.sha256((h_tr + h_te).encode()).hexdigest()
    run = {
        "config": cfg,
        "processed_data_sha256": {"train.csv": h_tr, "test.csv": h_te, "combined": combined},
        "ingestion": ingest_info,
        "validation": {"fail_checks": [c for c in val["checks"] if c["status"] == "FAIL"],
                       "rows_with_any_missing": val["rows_with_any_missing"],
                       "exact_duplicates": val["duplicates_exact"],
                       "label_distribution": val["label_distribution"]},
        "cleaning": clean_info,
        "feature_evaluation": feat.to_dict(orient="records"),
        "label_quality": labels,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    with open(results_dir / "run_config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(run, f, sort_keys=False, allow_unicode=True)
    (results_dir / "processed_data.sha256").write_text(
        f"{h_tr}  data/processed/train.csv\n{h_te}  data/processed/test.csv\n{combined}  combined\n",
        encoding="utf-8")
    log.info(f"HASH processed data sha256 (combined) = {combined}")
    log.info("DONE. Evidence in results/: validation_report.md, feature_evaluation.md, label_quality.md, "
             "preprocessor.joblib, run_config.yaml, processed_data.sha256, pipeline.log")


if __name__ == "__main__":
    main()
