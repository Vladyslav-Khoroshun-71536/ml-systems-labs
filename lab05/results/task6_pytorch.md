# Task 6: PyTorch internals

## 6a. Autograd graph of the 2-8-1 model (`loss.grad_fn` -> `next_functions`)

```
BinaryCrossEntropyBackward0
    SigmoidBackward0
        AddmmBackward0
            AccumulateGrad
            ReluBackward0
                AddmmBackward0
                    AccumulateGrad
                    TBackward0
                        AccumulateGrad
            TBackward0
                AccumulateGrad
```

`AccumulateGrad` leaves are the parameters (weights/biases): this is where `.grad` is accumulated, the same role as `+=` in our engine. `AddmmBackward0` = Linear layer (matrix multiply + bias), `TBackward0` = weight transpose.

## 6b. Eager vs compiled inference

`torch.no_grad()`, 20 warm-up calls, then the median of 100 calls. Compile/trace time is the one-off cost of the first call.

| model | input | mode | eager (ms) | compiled (ms) | speed-up | compile/trace time (s) |
|---|---|---|---|---|---|---|
| tiny MLP 2-8-1 | 200x2 | torch.jit.trace | 0.028 | 0.040 | 0.71x | 57.35 |
| MLP 512-1024-1024-10 | 256x512 | torch.jit.trace | 4.020 | 4.110 | 0.98x | 0.27 |
| many small ops (elementwise chain) | 256x1024 | torch.jit.trace | 3.378 | 3.407 | 0.99x | 0.47 |

Notes:

- tiny MLP 2-8-1: torch.compile failed: InductorError: InvalidCxxCompiler: Compiler: cl is not found.
- MLP 512-1024-1024-10: torch.compile failed: InductorError: InvalidCxxCompiler: Compiler: cl is not found.
- many small ops (elementwise chain): torch.compile failed: InductorError: InvalidCxxCompiler: Compiler: cl is not found.

## 6c. Shape, dtype, stride, contiguity

| tensor | shape | dtype | stride | contiguous | shares memory with A |
|---|---|---|---|---|---|
| A | (2048, 2048) | float32 | (2048, 1) | True | True |
| A.t() | (2048, 2048) | float32 | (1, 2048) | False | True |
| A[:, ::2] | (2048, 1024) | float32 | (2048, 2) | False | True |
| A[::2, :] | (1024, 2048) | float32 | (4096, 1) | False | True |
| A.t().contiguous() | (2048, 2048) | float32 | (2048, 1) | True | False |

`A.t()` and slicing return **views**: same memory, different strides. Only `.contiguous()` copies the data into row-major order.

## 6d. Cost of a non-contiguous view

Median of 20 runs after 3 warm-ups.

| operation (2048x2048 float32) | non-contiguous A.t() (ms) | contiguous copy (ms) | slow-down |
|---|---|---|---|
| elementwise y = x * 2 + 1 | 1.77 | 1.95 | 0.91x |
| row sum x.sum(dim=1) | 0.70 | 0.21 | 3.37x |
| matmul x @ B | 60.92 | 62.04 | 0.98x |

Making the copy itself (`A.t().contiguous()`) costs 7.08 ms.
