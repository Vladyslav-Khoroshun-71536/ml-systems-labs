import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (RESULTS, SEED, accuracy, backward, cross_entropy, forward,
                    init_params, load_data, set_seeds, sgd_step)

LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32


def train(width, lr, X_tr, Y_tr, y_tr, X_te, Y_te, y_te):
    params = init_params(width)
    rng = np.random.default_rng(SEED)
    hist = {"train_loss": [], "test_loss": [], "test_acc": []}
    n = X_tr.shape[0]
    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            P, cache = forward(params, X_tr[idx])
            batch_losses.append(cross_entropy(P, Y_tr[idx]) * len(idx))
            sgd_step(params, backward(params, cache, Y_tr[idx]), lr)
        train_loss = sum(batch_losses) / n
        P_te, _ = forward(params, X_te)
        test_loss = cross_entropy(P_te, Y_te)
        test_acc = float((P_te.argmax(axis=1) == y_te).mean())
        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["test_acc"].append(test_acc)
        if epoch == 1 or epoch % 10 == 0:
            print(f"  W={width:<3} lr={lr:<6} epoch {epoch:>2}: train loss {train_loss:.4f}"
                  f"  test loss {test_loss:.4f}  test acc {test_acc:.4f}")
    hist["final_train_acc"] = accuracy(params, X_tr, y_tr)
    hist["final_test_acc"] = hist["test_acc"][-1]
    return hist


def main() -> None:
    set_seeds()
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
    results = {}
    for width in WIDTHS:
        for lr in LRS:
            print(f"Training W={width}, lr={lr}")
            results[f"W{width}_lr{lr}"] = train(width, lr, X_tr, Y_tr, y_tr, X_te, Y_te, y_te)

    # loss curves: one panel per width, one line per learning rate
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    epochs = np.arange(1, EPOCHS + 1)
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for ax, width in zip(axes, WIDTHS):
        for lr in LRS:
            h = results[f"W{width}_lr{lr}"]
            ax.plot(epochs, h["train_loss"], color=colors[lr], label=f"lr={lr} train")
            ax.plot(epochs, h["test_loss"], color=colors[lr], linestyle="--", label=f"lr={lr} test")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("load_digits, 64-W-W-10 MLP from scratch, SGD batch 32")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)

    with open(RESULTS / "training_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["width", "lr", "final_train_loss", "final_test_loss", "final_train_acc", "final_test_acc"])
        for width in WIDTHS:
            for lr in LRS:
                h = results[f"W{width}_lr{lr}"]
                w.writerow([width, lr, f"{h['train_loss'][-1]:.4f}", f"{h['test_loss'][-1]:.4f}",
                            f"{h['final_train_acc']:.4f}", f"{h['final_test_acc']:.4f}"])
    (RESULTS / "training_history.json").write_text(json.dumps(results), encoding="utf-8")

    print("\nFinal test accuracy:")
    for width in WIDTHS:
        for lr in LRS:
            print(f"  W={width:<3} lr={lr:<6} {results[f'W{width}_lr{lr}']['final_test_acc']:.4f}")
    print(f"Saved {RESULTS / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
