import math

class Value:
    __slots__ = ("data", "grad", "parents", "op", "_backward", "label")

    def __init__(self, data, parents=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op
        self._backward = _no_grad
        self.label = label

    def __repr__(self):
        name = f"{self.label}=" if self.label else ""
        return f"Value({name}{self.data:.6g}, grad={self.grad:.6g})"

    def __add__(self, other):
        other = _lift(other)
        out = Value(self.data + other.data, (self, other), "+")

        def rule():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = rule
        return out

    def __mul__(self, other):
        other = _lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def rule():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = rule
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant int/float exponents are supported")
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def rule():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = rule
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def rule():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = rule
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def rule():
            self.grad += out.data * out.grad
        out._backward = rule
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def rule():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = rule
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def rule():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = rule
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_lift(other))

    def __truediv__(self, other):
        return self * _lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return _lift(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return _lift(other) / self

    def topological_order(self):
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                if id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self):
        order = self.topological_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

def _no_grad():
    pass

def _lift(x):
    return x if isinstance(x, Value) else Value(x)

def vsum(values):
    total = Value(0.0)
    for v in values:
        total = total + v
    return total

class Neuron:
    def __init__(self, n_in, activation, rng):
        bound = 1.0 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-bound, bound)) for _ in range(n_in)]
        self.b = Value(rng.uniform(-bound, bound))
        self.activation = activation

    def __call__(self, x):
        z = vsum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if self.activation == "relu":
            return z.relu()
        if self.activation == "sigmoid":
            return z.sigmoid()
        return z

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, activation, rng):
        self.neurons = [Neuron(n_in, activation, rng) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes=(2, 8, 1), rng=None):
        import random
        rng = rng or random.Random(0)
        acts = ["relu"] * (len(sizes) - 2) + ["sigmoid"]
        self.layers = [Layer(a, b, act, rng) for a, b, act in zip(sizes[:-1], sizes[1:], acts)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
