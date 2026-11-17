# Task 5: tiny model on make_moons

Data: `make_moons(n_samples=200, noise=0.15)`, standardised. Model: 2 -> 8 (ReLU) -> 1 (sigmoid), 33 parameters, binary cross-entropy, full-batch gradient descent, lr = 0.5, 100 epochs. Both frameworks start from **the same initial weights**. Time = median of 5 complete training runs.

| framework | median time (s) | per epoch (ms) | final loss | train acc |
|---|---|---|---|---|
| our engine (Value) | 11.310 | 113.10 | 0.257080 | 0.875 |
| PyTorch (eager, float64) | 0.0453 | 0.453 | 0.257080 | 0.875 |

- **Speed-up of PyTorch: 250x**
- Max |loss difference| over all 100 epochs: 3.33e-16 (same trajectory, so the engine's gradients are correct inside a real training loop too)
- Engine graph size per epoch: **19,036 Value nodes** (one Python object per scalar operation).
