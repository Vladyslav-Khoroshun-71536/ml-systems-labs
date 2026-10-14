import csv
import statistics
import time
import numpy as np

from common import RESULTS, backward, forward, init_params, load_data, set_seeds

RUNS = 100
WIDTHS = [16, 64]
BATCHES = [32, 540]


def matmul_macs(batch: int, w: int) -> tuple[int, int]:
    fwd = batch * 64 * w + batch * w * w + batch * w * 10
    bwd = (64 * batch * w
           + w * batch * w
           + w * batch * 10
           + batch * 10 * w
           + batch * w * w)
    return fwd, bwd


def median_us(fn) -> float:
    fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e6


def main() -> None:
    set_seeds()
    X_tr, X_te, Y_tr, Y_te, _, _ = load_data()
    rows = []
    for batch in BATCHES:
        X = X_tr[:batch] if batch <= X_tr.shape[0] else X_tr
        Y = Y_tr[:batch]
        for width in WIDTHS:
            params = init_params(width)
            _, cache = forward(params, X)
            t_fwd = median_us(lambda: forward(params, X))
            t_bwd = median_us(lambda: backward(params, cache, Y))
            macs_f, macs_b = matmul_macs(batch, width)
            rows.append([batch, width, t_fwd, t_bwd, t_bwd / t_fwd, macs_f, macs_b, macs_b / macs_f])

    header = ["batch", "width", "forward_us", "backward_us", "bwd/fwd time",
              "forward_matmul_MACs", "backward_matmul_MACs", "bwd/fwd MACs"]
    with open(RESULTS / "timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.2f}", r[5], r[6], f"{r[7]:.2f}"])

    print(f"Median of {RUNS} runs (microseconds)")
    print(f"{'batch':>5} {'W':>3} {'forward':>9} {'backward':>9} {'bwd/fwd':>8} {'MACs fwd':>9} {'MACs bwd':>9} {'ratio':>6}")
    for r in rows:
        print(f"{r[0]:>5} {r[1]:>3} {r[2]:>9.1f} {r[3]:>9.1f} {r[4]:>8.2f} {r[5]:>9} {r[6]:>9} {r[7]:>6.2f}")


if __name__ == "__main__":
    main()
