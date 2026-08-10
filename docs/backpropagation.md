# How Backpropagation Works

Backpropagation is reverse-mode automatic differentiation applied to a neural
network. NanoNet makes that process explicit.

## Computational graphs

A neural network is a composition of elementary operations. For a linear unit:

```text
y = (x * w) + b
```

the graph is:

```text
x ──► * ──► + ──► y
      ▲     ▲
      w     b
```

## Local derivatives

Each operation has a simple local derivative:

| Operation | Local derivative |
|-----------|------------------|
| `z = x * w` | `∂z/∂x = w`, `∂z/∂w = x` |
| `y = z + b` | `∂y/∂z = 1`, `∂y/∂b = 1` |

For the full expression `y = (x * w) + b`:

```text
∂y/∂x = w
∂y/∂w = x
∂y/∂b = 1
```

## The chain rule

Training minimizes a scalar loss `L`. If `L` depends on `y`, then:

```text
∂L/∂x = ∂L/∂y · ∂y/∂x
∂L/∂w = ∂L/∂y · ∂y/∂w
∂L/∂b = ∂L/∂y · ∂y/∂b
```

The upstream gradient `∂L/∂y` multiplies each local derivative. That is the
entire idea of backpropagation.

## Forward pass

1. Sample a batch of inputs.
2. Evaluate each layer left-to-right.
3. Compute the scalar loss.

The forward pass produces predictions **and** builds the computational graph
NanoNet needs for backward.

## Backward pass

1. Seed `∂L/∂L = 1`.
2. Walk the graph in reverse topological order.
3. At each node, apply the local backward rule.
4. Accumulate gradients into each Parameter.

## Gradient descent

Once gradients are known, an optimizer updates parameters. Plain SGD:

```text
θ ← θ − η ∇_θ L
```

Larger models use momentum or Adam, but the gradients themselves still come
from the same reverse-mode procedure.

## Dense layer view

For `Y = X @ W + b`:

```text
∂L/∂X = (∂L/∂Y) @ Wᵀ
∂L/∂W = Xᵀ @ (∂L/∂Y)
∂L/∂b = sum(∂L/∂Y, axis=batch)
```

NanoNet's `MatMul` and broadcasted `Add` implement exactly these identities.
