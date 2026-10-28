import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SEED = 42

def get_loaders(batch_size=128, data_dir=ROOT / "data"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, batch_size),
            DataLoader(test, batch_size))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, num_layers=2):
        super().__init__()
        self.patch = patch
        self.n_patches = (28 // patch) ** 2
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def patchify(self, x):
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        return x.reshape(x.size(0), -1, p * p)

    def forward(self, x):
        z = self.proj(self.patchify(x)) + self.pos     # (B,16,64)
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}
def linear_flops(i, o, n=1):
    return 2 * i * o * n


def conv_flops(k, cin, cout, hout, wout):
    return 2 * k * k * cin * cout * hout * wout


def attention_layer_flops(seq=16, d=64, ff=128):
    parts = {
        "QKV projections": 2 * seq * d * 3 * d,
        "scores QK^T": 2 * seq * seq * d,
        "weights x V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "feed-forward": 2 * seq * d * ff * 2,
    }
    parts["total"] = sum(parts.values())
    return parts


def hand_flops():
    att = attention_layer_flops()
    return {
        "one_layer": {
            "Linear 784->256": linear_flops(784, 256),
            "Conv2d(16,32,3) @14x14": conv_flops(3, 16, 32, 14, 14),
            "Transformer encoder layer (seq=16,d=64,ff=128)": att,
        },
        "model": {
            "MLP": linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10),
            "CNN": conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14) + linear_flops(1568, 10),
            "Attention": linear_flops(49, 64, 16) + 2 * att["total"] + linear_flops(64, 10),
        },
    }

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def torchinfo_summary(name, model):
    from torchinfo import summary
    s = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                col_names=("output_size", "num_params", "mult_adds"))
    (RESULTS / f"torchinfo_{name}.txt").write_text(str(s), encoding="utf-8")
    return s.total_params


def thop_macs(model):
    import copy
    from thop import profile
    m = copy.deepcopy(model).eval()
    macs, _ = profile(m, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    return int(macs)


def torch_flop_counter(model):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    fast = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(torch.randn(1, 1, 28, 28))
        return int(fc.get_total_flops())
    finally:
        if fast is not None:
            torch.backends.mha.set_fastpath_enabled(fast)


@torch.no_grad()
def latency_ms(model, batch, runs, warmup=10):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def activation_elems(model):
    total = 0

    def hook(_, __, out):
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel()

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    with torch.no_grad():
        model.eval()(torch.randn(1, 1, 28, 28))
    for h in hs:
        h.remove()
    return total

def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    return correct / n

def train(model, train_loader, val_loader, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.numel()
        val = evaluate(model, val_loader)
        history.append({"epoch": ep, "train_loss": total / len(train_loader.dataset),
                        "val_acc": val, "seconds": time.perf_counter() - t0})
        print(f"  epoch {ep}: loss {history[-1]['train_loss']:.4f}  val acc {val:.4f}  "
              f"({history[-1]['seconds']:.0f}s)")
    return history


def plot(res):
    names = list(res)
    colors = {"MLP": "#1f77b4", "CNN": "#2ca02c", "Attention": "#d62728"}
    acc = [res[n]["test_acc"] * 100 for n in names]
    for key, xlabel, fname, logx in [
        ("params", "Parameters", "acc_vs_params.png", True),
        ("latency_b1_ms", "Latency, batch 1 (ms, median of 100)", "acc_vs_latency.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        xs = [res[n][key] for n in names]
        for n, x, a in zip(names, xs, acc):
            ax.scatter(x, a, s=90, color=colors[n], zorder=3)
            ax.annotate(n, (x, a), textcoords="offset points", xytext=(8, -4))
        if key == "latency_b1_ms":
            for n in names:
                ax.scatter(res[n]["latency_b128_ms"] / 128, res[n]["test_acc"] * 100,
                           s=60, facecolors="none", edgecolors=colors[n], zorder=3)
            ax.set_xlabel("Latency (ms) — filled: batch 1; hollow: batch 128 per image")
        else:
            ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Fashion-MNIST: accuracy vs {key.split('_')[0]}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)


def write_summary(res, hand, threads):
    lines = ["# Measured results (auto-generated by lab3.py)", "",
             f"CPU threads: {threads}; torch {torch.__version__}", "",
             "| Model | Params | Hand FLOPs | thop MACs | thop x2 | torch FlopCounter | "
             "Latency b=1 (ms) | Latency b=128 (ms) | Activations/img | Test acc |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in res.items():
        fc = f"{r['flop_counter']:,}" if r["flop_counter"] else "n/a"
        lines.append(f"| {n} | {r['params']:,} | {hand['model'][n]:,} | {r['thop_macs']:,} | "
                     f"{2 * r['thop_macs']:,} | {fc} | {r['latency_b1_ms']:.3f} | "
                     f"{r['latency_b128_ms']:.2f} | {r['activation_elems']:,} | {r['test_acc'] * 100:.2f}% |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.threads)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders()
    hand = hand_flops()
    res = {}
    for name, cls in MODELS.items():
        print(f"\n=== {name} ===")
        torch.manual_seed(SEED)
        model = cls()
        r = {"params": count_params(model), "torchinfo_params": torchinfo_summary(name, model),
             "hand_flops": hand["model"][name], "thop_macs": thop_macs(model),
             "flop_counter": torch_flop_counter(model), "activation_elems": activation_elems(model),
             "latency_b1_ms": latency_ms(model, 1, runs=100),
             "latency_b128_ms": latency_ms(model, 128, runs=20)}
        print(f"  params {r['params']:,} | hand FLOPs {r['hand_flops']:,} | thop MACs {r['thop_macs']:,} "
              f"| FlopCounter {r['flop_counter']} | lat b1 {r['latency_b1_ms']:.3f} ms "
              f"| lat b128 {r['latency_b128_ms']:.2f} ms")
        r["history"] = train(model, train_loader, val_loader, epochs=args.epochs)
        r["test_acc"] = evaluate(model, test_loader)
        print(f"  TEST ACCURACY: {r['test_acc'] * 100:.2f}%")
        res[name] = r

    (RESULTS / "results.json").write_text(json.dumps({"hand_flops": hand, "models": res}, indent=2))
    write_summary(res, hand, args.threads)
    plot(res)
    print("\nSaved results/ and figures/.")


if __name__ == "__main__":
    main()
