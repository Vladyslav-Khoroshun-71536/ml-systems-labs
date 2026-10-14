"""Lab 01, steps 2-4: two baselines on Breast Cancer Wisconsin, system cost, deployment budgets.

Outputs (lab01/results/):
    baseline_accuracy.csv   test accuracy, 4 decimals            (step 2)
    measurements.csv        time, latency, size, peak memory     (step 3)
    model_logreg.joblib, model_rf.joblib                          (step 3, model size)
    budget_fit.csv          which deployment tier each model fits (step 4)
"""
from __future__ import annotations

import csv
import gc
import os
import random
import statistics
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_RUNS = 5        # after 1 warm-up, report the median
INFER_RUNS = 100      # single-sample inference, report the median

# Step 4 budgets, exactly as in the manual. Memory/size in MB, latency in ms.
# A model fits a tier when its need is within the tier's upper limit.
BUDGETS = {
    "Cloud":  {"memory_mb": None,          "latency_ms": 100, "size_mb": 500},   # memory >= 1 GB
    "Edge":   {"memory_mb": 1024,          "latency_ms": 50,  "size_mb": 50},    # 256-1024 MB
    "Mobile": {"memory_mb": 256,           "latency_ms": 20,  "size_mb": 10},    # 64-256 MB
    "TinyML": {"memory_mb": 256 / 1024,    "latency_ms": 10,  "size_mb": 100 / 1024},  # 256 KB, 100 KB
}


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:  # the manual asks for torch seeds too; torch is not used by these models
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def median_seconds(fn, runs: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def peak_mb(fn) -> float:
    """Peak resident memory of this process while fn runs (memory_profiler samples every 1 ms)."""
    gc.collect()
    return max(memory_usage((fn, (), {}), interval=0.001, max_usage=False, include_children=False))


def main() -> None:
    set_seeds()
    # The manual fixes LogisticRegression(max_iter=1000) on unscaled features; lbfgs then
    # stops at the iteration limit. We keep the required settings, silence the repeated
    # warning, and record whether it converged in measurements.csv.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # ---------------------------------------------------------------- step 2
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
    files = {"LogisticRegression": "model_logreg.joblib", "RandomForest": "model_rf.joblib"}

    baseline_rss = rss_mb()
    rows, acc_rows = [], []
    for name, proto in models.items():
        print(f"\n=== {name} ===")
        fitted = clone(proto).fit(X_tr, y_tr)
        acc = float((fitted.predict(X_te) == y_te).mean())
        acc_rows.append([name, f"{acc:.4f}"])
        print(f"Test accuracy: {acc:.4f}")

        # ------------------------------------------------------------ step 3
        train_s = median_seconds(lambda: clone(proto).fit(X_tr, y_tr), TRAIN_RUNS)
        sample = X_te[:1]
        infer_s = median_seconds(lambda: fitted.predict(sample), INFER_RUNS)

        path = RESULTS / files[name]
        joblib.dump(fitted, path)
        size_b = path.stat().st_size

        peak_train = peak_mb(lambda: clone(proto).fit(X_tr, y_tr))
        peak_infer = peak_mb(lambda: [fitted.predict(sample) for _ in range(INFER_RUNS)])

        n_iter = getattr(fitted, "n_iter_", None)
        converged = "n/a" if n_iter is None else bool(int(np.max(n_iter)) < proto.max_iter)
        row = {
            "model": name,
            "test_accuracy": round(acc, 4),
            "converged": converged,
            "train_time_ms_median": round(train_s * 1e3, 3),
            "inference_latency_ms_median": round(infer_s * 1e3, 4),
            "model_size_bytes": size_b,
            "model_size_kb": round(size_b / 1024, 2),
            "peak_rss_train_mb": round(peak_train, 1),
            "peak_rss_infer_mb": round(peak_infer, 1),
            "baseline_rss_mb": round(baseline_rss, 1),
            "train_rss_increase_mb": round(peak_train - baseline_rss, 1),
        }
        rows.append(row)
        for k, v in row.items():
            if k != "model":
                print(f"  {k}: {v}")

    with open(RESULTS / "baseline_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy"])
        w.writerows(acc_rows)

    with open(RESULTS / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- step 4
    fit_rows = []
    print("\n=== Deployment budget fit (memory = peak process RSS during inference) ===")
    for r in rows:
        need = {
            "memory_mb": r["peak_rss_infer_mb"],
            "latency_ms": r["inference_latency_ms_median"],
            "size_mb": r["model_size_bytes"] / 2**20,
        }
        for tier, b in BUDGETS.items():
            mem_ok = b["memory_mb"] is None or need["memory_mb"] <= b["memory_mb"]
            lat_ok = need["latency_ms"] <= b["latency_ms"]
            size_ok = need["size_mb"] <= b["size_mb"]
            fits = mem_ok and lat_ok and size_ok
            fit_rows.append([r["model"], tier, mem_ok, lat_ok, size_ok, fits])
            print(f"  {r['model']:<18} {tier:<7} memory {'ok' if mem_ok else 'NO':<3} "
                  f"latency {'ok' if lat_ok else 'NO':<3} size {'ok' if size_ok else 'NO':<3} -> "
                  f"{'FITS' if fits else 'does not fit'}")

    with open(RESULTS / "budget_fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "tier", "memory_ok", "latency_ok", "size_ok", "fits"])
        w.writerows(fit_rows)
    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()
