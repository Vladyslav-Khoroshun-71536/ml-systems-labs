import math
import numpy as np
from common import RESULTS, cross_entropy, forward, init_params, load_data, n_params, set_seeds

def main() -> None:
    set_seeds()
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
    lines = [
        f"Train: X {X_tr.shape}, Y {Y_tr.shape} | Test: X {X_te.shape}, Y {Y_te.shape}",
        f"Input range after /16: [{X_tr.min():.2f}, {X_tr.max():.2f}]",
        f"Class counts train: {np.bincount(y_tr).tolist()}",
        "",
    ]
    ok_all = True
    for width in (16, 64):
        params = init_params(width)
        shapes = ", ".join(f"{k}{v.shape}" for k, v in params.items())
        P, _ = forward(params, X_tr)
        row_err = float(np.abs(P.sum(axis=1) - 1.0).max())
        loss = cross_entropy(P, Y_tr)
        shape_ok = P.shape == (X_tr.shape[0], 10)
        sum_ok = row_err < 1e-6
        loss_ok = abs(loss - math.log(10)) < 0.05
        ok_all &= shape_ok and sum_ok and loss_ok
        lines += [
            f"W = {width}: {shapes}  -> {n_params(params)} parameters",
            f"  output shape {P.shape} (expected (batch, 10)): {'OK' if shape_ok else 'FAIL'}",
            f"  max |row sum - 1| = {row_err:.2e} (< 1e-6): {'OK' if sum_ok else 'FAIL'}",
            f"  untrained loss = {loss:.4f} (ln 10 = {math.log(10):.4f}): {'OK' if loss_ok else 'FAIL'}",
            "",
        ]
    lines.append("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "sanity_checks.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
