import csv
import numpy as np
from common import RESULTS, SEED, backward, cross_entropy, forward, init_params, load_data, set_seeds

H = 1e-5
THRESHOLD = 1e-6
N_WEIGHTS = 10
BATCH = 64


def loss_at(params, X, Y):
    P, _ = forward(params, X)
    return cross_entropy(P, Y)


def check(width: int, rng: np.random.Generator, X, Y):
    params = init_params(width)
    _, cache = forward(params, X)
    grads = backward(params, cache, Y)

    rows = []
    names = ["W1", "W2", "W3"]
    for _ in range(N_WEIGHTS):
        name = names[rng.integers(len(names))]
        idx = tuple(int(rng.integers(s)) for s in params[name].shape)
        old = params[name][idx]
        params[name][idx] = old + H
        l_plus = loss_at(params, X, Y)
        params[name][idx] = old - H
        l_minus = loss_at(params, X, Y)
        params[name][idx] = old
        numerical = (l_plus - l_minus) / (2 * H)
        analytic = grads[name][idx]
        rel = abs(analytic - numerical) / max(abs(analytic) + abs(numerical), 1e-12)
        rows.append((width, name, str(idx), analytic, numerical, rel, rel < THRESHOLD))
    return rows


def main() -> None:
    set_seeds()
    X_tr, _, Y_tr, _, _, _ = load_data()
    rng = np.random.default_rng(SEED)
    batch = rng.choice(X_tr.shape[0], size=BATCH, replace=False)
    X, Y = X_tr[batch], Y_tr[batch]

    all_rows, summary = [], []
    for width in (16, 64):
        rows = check(width, rng, X, Y)
        all_rows += rows
        passed = sum(r[-1] for r in rows)
        verdict = "PASS" if passed >= 9 else "FAIL"
        summary.append(f"W = {width}: {passed}/{N_WEIGHTS} weights with relative error < {THRESHOLD:g} -> {verdict}")

    with open(RESULTS / "gradient_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["width", "param", "index", "analytic", "numerical", "relative_error", "pass"])
        for r in all_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.10e}", f"{r[4]:.10e}", f"{r[5]:.3e}", r[6]])

    print(f"{'W':>3} {'param':>5} {'index':>9} {'analytic':>14} {'numerical':>14} {'rel.err':>9}")
    for r in all_rows:
        print(f"{r[0]:>3} {r[1]:>5} {r[2]:>9} {r[3]:>14.6e} {r[4]:>14.6e} {r[5]:>9.2e} {'ok' if r[6] else 'X'}")
    print("\n" + "\n".join(summary))
    (RESULTS / "gradient_check.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
