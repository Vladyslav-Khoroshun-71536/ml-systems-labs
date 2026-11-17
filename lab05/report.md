# Lab 5 — Frameworks and automatic differentiation

```bash
pip install -r requirements.txt
python lab5.py            # all tasks, about 2 minutes; results in results/
python lab5.py --quick    # short check, about 40 s
```

| File | Content |
|---|---|
| [`engine.py`](engine.py) | Tasks 1–3: `Value` node, operations, topological backward pass, tiny MLP built from `Value`s |
| [`lab5.py`](lab5.py) | Tasks 4–6: gradient check, make_moons benchmark, PyTorch graph / compile / contiguity |
| `results/task4_gradient_check.md` | Engine vs PyTorch vs numerical gradients |
| `results/task5_moons.md` | Training speed, engine vs PyTorch |
| `results/task6_pytorch.md` | Autograd graph, eager vs compiled, tensor strides, non-contiguous timing |
| `results/results.json` | All numbers and the environment (Python, torch, CPU) |

The measured numbers below are from my run: Linux, 2 CPU threads, PyTorch 2.15. Timings change between machines, so the files in `results/` hold the values from your own run.

## 1–3. The engine

`Value` stores four things:

- `data` — the number;
- `grad` — the gradient, i.e. d(output)/d(this value);
- `parents` — the nodes it was computed from;
- `_backward` — a closure with the **local derivative rule**.

Each operation creates a new node, so the forward pass itself builds the computation graph. These are the local rules (chain rule, `out.grad` is the incoming gradient):

| Operation | Rule in `_backward` |
|---|---|
| `a + b` | `a.grad += out.grad`, `b.grad += out.grad` |
| `a * b` | `a.grad += b·out.grad`, `b.grad += a·out.grad` |
| `a ** n` | `a.grad += n·aⁿ⁻¹·out.grad` |
| `relu(a)` | `a.grad += [a > 0]·out.grad` |
| `exp`, `log`, `sigmoid` (for task 5) | `eᵃ`, `1/a`, `σ(1−σ)` |

Subtraction, negation and division are built from `+`, `*` and `**`, so they need no rules of their own.

`backward()` does three steps:

1. **Topological sort** (iterative DFS, no recursion limit): every node comes after all its parents.
2. Set **output grad = 1**.
3. Visit the nodes **in reverse order** and call each `_backward`.

Gradients are **accumulated with `+=`**. If a value feeds into several operations, its gradient is the sum over all paths. With `=` the last path would overwrite the others. Test: `y = x*x + x` at x = 3 gives dy/dx = 2x + 1 = **7** ✓.

## 4. Gradient check

The function `((a*b + c)**2).relu().sum()` is evaluated on 6-element float64 vectors. One element sits exactly at the ReLU kink (a·b + c = 0).

| Comparison | Max absolute difference | Requirement |
|---|---:|---|
| Engine vs PyTorch autograd | **0.0** (bit-identical) | < 1e-6 ✓ |
| Engine vs numerical central differences (ε = 1e-6) | **1.05e-9** | < 1e-6 ✓ |

The numerical gradient differs by about 1e-9. This is the truncation and rounding error of finite differences, not an engine error. float64 is used because in float32 the rounding alone would be close to 1e-6.

## 5. make_moons: engine vs PyTorch

Setup:

- 200 points, model 2 → 8 (ReLU) → 1 (sigmoid), 33 parameters;
- BCE loss, full-batch gradient descent, lr = 0.5, 100 epochs;
- **both frameworks start from the same weights**;
- time is the median of 5 full training runs.

| | Median time | Per epoch | Final loss | Train acc |
|---|---:|---:|---:|---:|
| Our engine | 19.1 s | 191 ms | 0.257080 | 87.5 % |
| PyTorch eager | 0.035 s | 0.35 ms | 0.257080 | 87.5 % |

**PyTorch is ≈ 540× faster**, and the loss curves are identical (max difference 3e-16). This proves the engine's gradients are also right inside a real training loop.

Why so slow:

- One epoch builds **19,036 `Value` objects**, each with a Python closure.
- Every scalar multiply-add costs a Python function call and a memory allocation.
- PyTorch performs the same work as a few matrix operations (`addmm`) in optimized C++. It processes all 200 samples at once and builds a graph with only ~10 nodes.

This is the **framework overhead** that Chapter 7 discusses: graph size and per-op dispatch cost dominate when operations are tiny.

The first PyTorch run took 1.0 s instead of 0.035 s. That is one-off initialisation (kernel dispatch tables, thread pool), which is why the median is used.

## 6. PyTorch internals

**Graph.** `loss.grad_fn` → `next_functions` gives:

`BinaryCrossEntropyBackward0 → SigmoidBackward0 → AddmmBackward0 → ReluBackward0 → AddmmBackward0 → … → AccumulateGrad`

- `AddmmBackward0` is a Linear layer (matrix multiply + bias).
- `AccumulateGrad` leaves are the parameters, where `.grad` is accumulated, exactly like `+=` in the engine.

The structure is the same as ours, but with one node per *tensor* operation instead of one per *scalar*.

**Eager vs compiled.** Inference only, after 20 warm-up calls, median of 100 calls:

| Model | Eager | `torch.compile` | Speed-up | One-off compile time |
|---|---:|---:|---:|---:|
| Tiny MLP 2-8-1, batch 200 | 0.016 ms | 0.034 ms | 0.48× (slower) | 23 s |
| MLP 512-1024-1024-10, batch 256 | 4.28 ms | 4.48 ms | 0.96× | 1 s |
| 10× elementwise chain, 256×1024 | 4.20 ms | 5.10 ms | 0.82× | 1 s |

On this 2-thread CPU, compilation **did not help**:

- **Tiny model.** The work is microseconds, so the compiled wrapper's guard checks cost more than they save.
- **Large MLP.** The time is dominated by the matrix multiplies, and those already call the same optimized BLAS kernels in eager mode. There is nothing left to optimize.
- **Elementwise chain.** This is where operator fusion should help (one loop over memory instead of ~50). On a 2-core CPU the generated kernel was not faster than PyTorch's vectorized eager kernels.

On Windows, `torch.compile` may fail (it needs a C++ compiler). The script then falls back to `torch.jit.trace` and notes this in `task6_pytorch.md`.

**Tensors, strides, contiguity.** For a 2048 × 2048 float32 tensor, `A.t()`, `A[:, ::2]` and `A[::2, :]` are **views**: they share A's memory and only change the strides, so `is_contiguous()` returns False. Only `.contiguous()` copies the data, which costs ≈ 5 ms here.

| Operation | On view `A.t()` | On contiguous copy | Effect |
|---|---:|---:|---:|
| `x * 2 + 1` | 1.61 ms | 1.61 ms | none |
| `x.sum(dim=1)` | 0.71 ms | 0.40 ms | **1.8× slower** |
| `x @ B` | 67.2 ms | 65.6 ms | none |

The cost depends on the operation:

- **Reduction along rows.** On the transposed view, a row is read with a stride of 2048 floats, so every element comes from a different cache line. This is the strided access pattern from Reddi §4.7 (Lab 3).
- **Elementwise op.** PyTorch's TensorIterator reorders the loop to walk memory in its real order, so there is no penalty.
- **matmul.** BLAS just receives a "transposed" flag.

## When compilation helps

`torch.compile` / tracing pays off when **all** of the following hold:

1. The model is **called many times with the same input shapes**. Compilation costs seconds, and a new shape triggers recompilation.
2. The run time is dominated by **many small, fusable operations** (elementwise chains, normalisation, activations), not by one big matmul that BLAS or cuDNN already handles well.
3. Per-operation **Python/dispatch overhead is significant** relative to the arithmetic, e.g. a medium model on a GPU, where launching 50 kernels costs more than running one fused kernel.
4. There is enough hardware parallelism for the generated kernels to beat the eager ones. The biggest gains are usually on GPUs.

It does **not** help for tiny models, one-off scripts, constantly changing shapes, matmul-bound models on CPU, or code with data-dependent Python control flow, where graph breaks fall back to eager execution. **Measure after warm-up and count the compile time.** Our tiny MLP became slower after compiling, so its 23 s compile time would never be recovered.
