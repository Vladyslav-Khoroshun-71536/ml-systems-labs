from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INIT_STD = 0.1


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_data():

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    return X_tr, X_te, one_hot(y_tr), one_hot(y_te), y_tr, y_te


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def init_params(width: int, std: float = INIT_STD, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, std, size=(64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0.0, std, size=(width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0.0, std, size=(width, 10)),
        "b3": np.zeros(10),
    }


def n_params(params: dict) -> int:
    return sum(v.size for v in params.values())

def relu(z):
    return np.maximum(z, 0.0)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params: dict, X: np.ndarray):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    P = softmax(Z3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "P": P}
    return P, cache

def cross_entropy(P: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.sum(Y * np.log(P + eps)) / Y.shape[0])

def backward(params: dict, cache: dict, Y: np.ndarray) -> dict:

    N = Y.shape[0]
    dZ3 = (cache["P"] - Y) / N
    dW3 = cache["A2"].T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    P, _ = forward(params, X)
    return float((P.argmax(axis=1) == y).mean())
