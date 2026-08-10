# Automatic Differentiation

NanoNet implements **reverse-mode automatic differentiation** (backpropagation
for scalar outputs). Every Tensor operation that needs gradients records a
small `Function` node. Together, those nodes form a computational graph.

## Example graph

For:

```python
x = Tensor(2.0, requires_grad=True)
a = x * x
b = x * 3
y = a + b
```

the graph looks like:

```text
      x
     / \
    *   *
   /     \
  x       3
   \     /
      +
      |
      y
```

`x` influences `y` through two paths, so its gradient is the **sum** of both
path contributions:

```text
∂y/∂x = ∂y/∂a · ∂a/∂x + ∂y/∂b · ∂b/∂x
      = 1 · 2x + 1 · 3
      = 2x + 3
```

At `x = 2`, the gradient is `7`.

## Topological order

Before backward, NanoNet builds a topological ordering of tensors that
participate in the graph. Parents are visited before children, so the reverse
of that list is a valid backward schedule: every node receives its upstream
gradient before distributing local gradients to its parents.

Why reverse order? The chain rule needs the upstream gradient first. You cannot
compute `∂L/∂x` for an intermediate node until you know `∂L/∂y` for every
downstream consumer of that node.

## Gradient accumulation

If a tensor is reused, NanoNet **adds** into `.grad` rather than overwriting it.
That is required for branching graphs and for calling `backward()` more than
once without clearing gradients.

`zero_grad()` sets `.grad` to `None`, meaning "no gradient accumulated yet."

## Broadcasting

When shapes differ, NumPy broadcasts during the forward pass. During backward,
gradients must be reduced with `unbroadcast` so each parameter receives a
gradient matching its own shape. For example, a bias of shape `(128,)` added to
a batch of shape `(32, 128)` receives the sum of the upstream gradient over the
batch axis.

## Graph lifetime

After `backward()`, NanoNet does not keep an explicit `retain_graph` API.
Intermediate Function nodes become unreachable and can be garbage-collected.
Build a fresh forward graph for each training step (the normal pattern).
