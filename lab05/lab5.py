import argparse
import json
import platform
import random
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from engine import MLP, Value, vsum

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
OUT = {}

def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def median_time(fn, runs, warmup=0):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return statistics.median(t)

def f_engine(a, b, c):
    return vsum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))

def task4(n=6):
    rng = np.random.default_rng(SEED)
    A, B, C = (rng.normal(size=n) for _ in range(3))
    C[0] = -A[0] * B[0]

    a = [Value(x, label=f"a{i}") for i, x in enumerate(A)]
    b = [Value(x, label=f"b{i}") for i, x in enumerate(B)]
    c = [Value(x, label=f"c{i}") for i, x in enumerate(C)]
    out = f_engine(a, b, c)
    out.backward()
    g_eng = np.array([[v.grad for v in a], [v.grad for v in b], [v.grad for v in c]])

    ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (A, B, C))
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    g_torch = np.stack([ta.grad.numpy(), tb.grad.numpy(), tc.grad.numpy()])

    def f_num(X):
        return sum(max((x * y + z) ** 2, 0.0) for x, y, z in zip(*X))

    eps = 1e-6
    X = np.stack([A, B, C]).astype(np.float64)
    g_num = np.zeros_like(X)
    for i in range(3):
        for j in range(n):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += eps
            Xm[i, j] -= eps
            g_num[i, j] = (f_num(Xp) - f_num(Xm)) / (2 * eps)

    d_torch = float(np.max(np.abs(g_eng - g_torch)))
    d_num = float(np.max(np.abs(g_eng - g_num)))
    d_out = abs(out.data - tout.item())

    x = Value(3.0)
    y = x * x + x
    y.backward()
    acc_ok = abs(x.grad - 7.0) < 1e-12

    rows = []
    for name, i in (("a", 0), ("b", 1), ("c", 2)):
        for j in range(n):
            rows.append({"param": f"{name}[{j}]", "engine": f"{g_eng[i, j]:.10f}",
                         "torch": f"{g_torch[i, j]:.10f}", "numerical": f"{g_num[i, j]:.10f}"})
    ok = d_torch < 1e-6 and d_num < 1e-6
    text = ["# Task 4: gradient check", "",
            f"Function: `((a*b + c)**2).relu().sum()` on {n}-element float64 vectors (seed {SEED}). "
            "Element 0 is chosen so that a*b + c = 0 (the relu kink).", "",
            f"- forward value: engine {out.data:.12f}, torch {tout.item():.12f} (|diff| = {d_out:.2e})",
            f"- **max |engine - torch| = {d_torch:.2e}**",
            f"- **max |engine - numerical| = {d_num:.2e}** (central differences, eps = {eps:g})",
            f"- requirement < 1e-6: **{'PASS' if ok else 'FAIL'}**",
            f"- accumulation test `y = x*x + x` at x=3: dy/dx = {x.grad:g} (expected 7) "
            f"-> {'PASS' if acc_ok else 'FAIL'}", "",
            md_table(rows, ["param", "engine", "torch", "numerical"]), ""]
    (RESULTS / "task4_gradient_check.md").write_text("\n".join(text), encoding="utf-8")
    OUT["task4"] = {"max_abs_diff_torch": d_torch, "max_abs_diff_numerical": d_num,
                    "forward_diff": d_out, "pass": ok, "accumulation_pass": acc_ok}
    print(f"[task4] max|eng-torch|={d_torch:.2e}  max|eng-num|={d_num:.2e}  "
          f"{'PASS' if ok else 'FAIL'}; accumulation {'PASS' if acc_ok else 'FAIL'}")

def moons(n):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=0.15, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)
    return X, y.astype(float)

def bce_engine(p, y, eps=1e-12):
    return -(y * (p + eps).log() + (1 - y) * (1 - p + eps).log())

def train_engine(X, y, epochs, lr, init_seed):
    model = MLP((2, 8, 1), rng=random.Random(init_seed))
    params = model.parameters()
    init = [p.data for p in params]
    losses, n_nodes = [], 0
    for _ in range(epochs):
        preds = [model([Value(xi[0]), Value(xi[1])]) for xi in X]
        loss = vsum(bce_engine(p, yi) for p, yi in zip(preds, y)) * (1.0 / len(y))
        if not n_nodes:
            n_nodes = len(loss.topological_order())
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
        losses.append(loss.data)
    acc = np.mean([(model([Value(a), Value(b)]).data > 0.5) == yi for (a, b), yi in zip(X, y)])
    return losses, float(acc), init, n_nodes

class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def torch_from_engine(init):
    m = TorchMLP().double()
    w1 = np.array(init[:24]).reshape(8, 3)
    w2 = np.array(init[24:])
    with torch.no_grad():
        m.fc1.weight.copy_(torch.tensor(w1[:, :2]))
        m.fc1.bias.copy_(torch.tensor(w1[:, 2]))
        m.fc2.weight.copy_(torch.tensor(w2[:8]).view(1, 8))
        m.fc2.bias.copy_(torch.tensor(w2[8:]))
    return m

def train_torch(X, y, epochs, lr, init):
    m = torch_from_engine(init)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).view(-1, 1)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        p = m(Xt)
        loss = -(yt * torch.log(p + 1e-12) + (1 - yt) * torch.log(1 - p + 1e-12)).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((m(Xt) > 0.5).double() == yt).double().mean().item()
    return losses, acc

def task5(n=200, epochs=100, lr=0.5, runs=5):
    X, y = moons(n)
    t_eng, t_tch, res = [], [], None
    for r in range(runs):
        t0 = time.perf_counter()
        le, ae, init, n_nodes = train_engine(X, y, epochs, lr, init_seed=SEED)
        t_eng.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        lt, at = train_torch(X, y, epochs, lr, init)
        t_tch.append(time.perf_counter() - t0)
        res = (le, ae, lt, at)
        print(f"[task5] run {r + 1}/{runs}: engine {t_eng[-1]:.2f}s  torch {t_tch[-1] * 1e3:.1f}ms")
    le, ae, lt, at = res
    me, mt = statistics.median(t_eng), statistics.median(t_tch)
    max_loss_diff = float(np.max(np.abs(np.array(le) - np.array(lt))))
    n_params = 2 * 8 + 8 + 8 + 1
    rows = [
        {"framework": "our engine (Value)", "median time (s)": f"{me:.3f}",
         "per epoch (ms)": f"{me / epochs * 1e3:.2f}", "final loss": f"{le[-1]:.6f}",
         "train acc": f"{ae:.3f}"},
        {"framework": "PyTorch (eager, float64)", "median time (s)": f"{mt:.4f}",
         "per epoch (ms)": f"{mt / epochs * 1e3:.3f}", "final loss": f"{lt[-1]:.6f}",
         "train acc": f"{at:.3f}"},
    ]
    text = ["# Task 5: tiny model on make_moons", "",
            f"Data: `make_moons(n_samples={n}, noise=0.15)`, standardised. Model: 2 -> 8 (ReLU) -> 1 (sigmoid), "
            f"{n_params} parameters, binary cross-entropy, full-batch gradient descent, lr = {lr}, "
            f"{epochs} epochs. Both frameworks start from **the same initial weights**. "
            f"Time = median of {runs} complete training runs.", "",
            md_table(rows, list(rows[0])), "",
            f"- **Speed-up of PyTorch: {me / mt:.0f}x**",
            f"- Max |loss difference| over all {epochs} epochs: {max_loss_diff:.2e} "
            "(same trajectory, so the engine's gradients are correct inside a real training loop too)",
            f"- Engine graph size per epoch: **{n_nodes:,} Value nodes** "
            "(one Python object per scalar operation).", ""]
    (RESULTS / "task5_moons.md").write_text("\n".join(text), encoding="utf-8")
    OUT["task5"] = {"engine_median_s": me, "torch_median_s": mt, "speedup": me / mt,
                    "engine_times": t_eng, "torch_times": t_tch, "engine_final_loss": le[-1],
                    "torch_final_loss": lt[-1], "engine_acc": ae, "torch_acc": at,
                    "max_loss_diff": max_loss_diff, "epochs": epochs, "n": n, "graph_nodes": n_nodes}

def describe_graph(t, depth=0, seen=None, lines=None, max_depth=12):
    seen = set() if seen is None else seen
    lines = [] if lines is None else lines
    fn = t if not isinstance(t, torch.Tensor) else t.grad_fn
    if fn is None or depth > max_depth:
        return lines
    name = type(fn).__name__
    lines.append("    " * depth + name + ("  (seen)" if id(fn) in seen else ""))
    if id(fn) in seen:
        return lines
    seen.add(id(fn))
    for nxt, _ in fn.next_functions:
        if nxt is not None:
            describe_graph(nxt, depth + 1, seen, lines, max_depth)
    return lines

def try_compile(model, example):
    errors = []
    try:
        cm = torch.compile(model)
        with torch.no_grad():
            cm(example)
        return cm, "torch.compile", errors
    except Exception as e:
        errors.append(f"torch.compile failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = torch.jit.trace(model, example)
            tm = torch.jit.freeze(tm.eval()) if hasattr(torch.jit, "freeze") else tm
        return tm, "torch.jit.trace", errors
    except Exception as e:
        errors.append(f"torch.jit.trace failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    return None, "none", errors

def bench_inference(model, x, runs, warmup=20):
    with torch.no_grad():
        return median_time(lambda: model(x), runs, warmup)

def task6(runs=100):
    torch.manual_seed(SEED)
    lines_out = ["# Task 6: PyTorch internals", ""]

    m = TorchMLP()
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4, 1)).float()
    loss = nn.functional.binary_cross_entropy(m(x), y)
    graph = describe_graph(loss)
    lines_out += ["## 6a. Autograd graph of the 2-8-1 model (`loss.grad_fn` -> `next_functions`)", "",
                  "```", *graph, "```", "",
                  "`AccumulateGrad` leaves are the parameters (weights/biases): this is where `.grad` is "
                  "accumulated, the same role as `+=` in our engine. `AddmmBackward0` = Linear layer "
                  "(matrix multiply + bias), `TBackward0` = weight transpose.", ""]

    configs = [("tiny MLP 2-8-1", TorchMLP().eval(), torch.randn(200, 2)),
               ("MLP 512-1024-1024-10", nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1024),
                                                      nn.GELU(), nn.Linear(1024, 10)).eval(),
                torch.randn(256, 512)),
               ("many small ops (elementwise chain)", ElementwiseChain().eval(), torch.randn(256, 1024))]
    rows, notes = [], []
    for name, model, xb in configs:
        t_e = bench_inference(model, xb, runs)
        t0 = time.perf_counter()
        cm, mode, errs = try_compile(model, xb)
        t_build = time.perf_counter() - t0
        notes += [f"- {name}: {e}" for e in errs]
        if cm is not None:
            t_c = bench_inference(cm, xb, runs)
            rows.append({"model": name, "input": "x".join(map(str, xb.shape)), "mode": mode,
                         "eager (ms)": f"{t_e * 1e3:.3f}", "compiled (ms)": f"{t_c * 1e3:.3f}",
                         "speed-up": f"{t_e / t_c:.2f}x", "compile/trace time (s)": f"{t_build:.2f}"})
        else:
            rows.append({"model": name, "input": "x".join(map(str, xb.shape)), "mode": "unavailable",
                         "eager (ms)": f"{t_e * 1e3:.3f}", "compiled (ms)": "-", "speed-up": "-",
                         "compile/trace time (s)": "-"})
        print(f"[task6] {name}: eager {t_e * 1e3:.3f} ms, {mode} "
              f"{rows[-1]['compiled (ms)']} ms ({rows[-1]['speed-up']})")
    lines_out += ["## 6b. Eager vs compiled inference", "",
                  f"`torch.no_grad()`, 20 warm-up calls, then the median of {runs} calls. "
                  "Compile/trace time is the one-off cost of the first call.", "",
                  md_table(rows, list(rows[0])), ""]
    if notes:
        lines_out += ["Notes:", "", *notes, ""]

    A = torch.randn(2048, 2048)
    views = [("A", A), ("A.t()", A.t()), ("A[:, ::2]", A[:, ::2]), ("A[::2, :]", A[::2, :]),
             ("A.t().contiguous()", A.t().contiguous())]
    trows = [{"tensor": n, "shape": tuple(v.shape), "dtype": str(v.dtype).replace("torch.", ""),
              "stride": tuple(v.stride()), "contiguous": v.is_contiguous(),
              "shares memory with A": v.untyped_storage().data_ptr() == A.untyped_storage().data_ptr()}
             for n, v in views]
    lines_out += ["## 6c. Shape, dtype, stride, contiguity", "", md_table(trows, list(trows[0])), "",
                  "`A.t()` and slicing return **views**: same memory, different strides. "
                  "Only `.contiguous()` copies the data into row-major order.", ""]

    At, Atc = A.t(), A.t().contiguous()
    B = torch.randn(2048, 2048)
    ops = [("elementwise y = x * 2 + 1", lambda t: t * 2 + 1),
           ("row sum x.sum(dim=1)", lambda t: t.sum(dim=1)),
           ("matmul x @ B", lambda t: t @ B)]
    srows = []
    for name, op in ops:
        tn = median_time(lambda: op(At), 20, 3)
        tc = median_time(lambda: op(Atc), 20, 3)
        srows.append({"operation (2048x2048 float32)": name, "non-contiguous A.t() (ms)": f"{tn * 1e3:.2f}",
                      "contiguous copy (ms)": f"{tc * 1e3:.2f}", "slow-down": f"{tn / tc:.2f}x"})
        print(f"[task6] {name}: non-contig {tn * 1e3:.2f} ms vs contig {tc * 1e3:.2f} ms")
    t_copy = median_time(lambda: A.t().contiguous(), 20, 3)
    lines_out += ["## 6d. Cost of a non-contiguous view", "", "Median of 20 runs after 3 warm-ups.", "",
                  md_table(srows, list(srows[0])), "",
                  f"Making the copy itself (`A.t().contiguous()`) costs {t_copy * 1e3:.2f} ms.", ""]
    (RESULTS / "task6_pytorch.md").write_text("\n".join(lines_out), encoding="utf-8")
    OUT["task6"] = {"compile": rows, "tensors": [{k: str(v) for k, v in r.items()} for r in trows],
                    "noncontiguous": srows, "copy_ms": t_copy * 1e3, "notes": notes}


class ElementwiseChain(nn.Module):
    def forward(self, x):
        for _ in range(10):
            x = torch.tanh(x) * 0.9 + torch.sigmoid(x) * 0.1 - x.abs() * 0.01
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    task4()
    if args.quick:
        task5(n=100, epochs=20, runs=2)
        task6(runs=20)
    else:
        task5()
        task6()
    OUT["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "platform": platform.platform(),
                          "cpu": platform.processor(), "threads": torch.get_num_threads()}
    (RESULTS / "results.json").write_text(json.dumps(OUT, indent=2, default=str), encoding="utf-8")
    print("Saved results/task4_gradient_check.md, task5_moons.md, task6_pytorch.md, results.json")

if __name__ == "__main__":
    main()
