# Automatic Differentiation

NanoNet implements **reverse-mode automatic differentiation**, the same general
form of differentiation used to train modern neural networks.

Rather than deriving the gradient of an entire neural network manually,
NanoNet records individual tensor operations during the forward pass and uses
their local derivatives during the backward pass.

---

## Computational Graphs

Every differentiable Tensor operation creates a small node in a computational
graph.

For example:

```python
from nanonet_ml import Tensor

x = Tensor(2.0, requires_grad=True)

a = x * x
b = x * 3
y = a + b
```

The corresponding graph is conceptually:

```text
        x
       / \
      /   \
     *     *
    / \   / \
   x   x x   3
    \     /
     \   /
       +
       |
       y
```

The same tensor can contribute to a result through multiple paths.

In this case:

```text
y = x² + 3x
```

so:

```text
dy/dx = 2x + 3
```

At:

```text
x = 2
```

the gradient is:

```text
dy/dx = 7
```

NanoNet automatically discovers and combines these gradient contributions.

---

## Reverse-Mode Automatic Differentiation

Neural-network training normally produces a single scalar loss from a large
number of parameters.

Reverse-mode automatic differentiation is especially efficient for this case
because it calculates the derivative of one output with respect to many
inputs in a single backward traversal.

A simplified training computation might look like:

```text
Input
  │
  ▼
Dense
  │
  ▼
ReLU
  │
  ▼
Dense
  │
  ▼
Loss
```

The forward pass moves downward through the graph.

The backward pass traverses the graph in the opposite direction:

```text
Input
  ▲
  │
Dense
  ▲
  │
ReLU
  ▲
  │
Dense
  ▲
  │
Loss
```

Each node receives an upstream gradient and applies its local derivative.

---

## Topological Ordering

Before performing the backward pass, NanoNet constructs a **topological
ordering** of the tensors involved in the graph.

Parents are visited before their dependent outputs.

The resulting order can then be reversed so that every operation receives its
upstream gradient before it needs to propagate gradients toward its own
inputs.

Conceptually:

```text
Forward dependency order:

x → a → b → y
```

becomes:

```text
Backward traversal:

y → b → a → x
```

For graphs with branches, a node can receive multiple gradient contributions.
Those contributions must be combined before or while propagating farther
backward.

---

## The Chain Rule

Suppose:

```text
y = f(x)
L = g(y)
```

Then:

```text
dL/dx = dL/dy × dy/dx
```

The incoming gradient:

```text
dL/dy
```

is often called the **upstream gradient**.

Each operation multiplies that upstream value by its own local derivative and
passes the result toward its parents.

This process continues until NanoNet reaches the leaf tensors and trainable
parameters.

---

## Gradient Accumulation

NanoNet accumulates gradients in each tensor's `.grad` attribute.

For example:

```python
from nanonet_ml import Tensor

x = Tensor(2.0, requires_grad=True)

y = x * x
y.backward()

print(x.grad)
```

produces:

```text
4.0
```

Calling `backward()` again on the same graph adds another independent gradient
contribution:

```python
y.backward()

print(x.grad)
```

produces:

```text
8.0
```

The second backward pass does **not** propagate the previously accumulated
gradient through the graph again.

NanoNet maintains a temporary per-backward-pass propagation buffer so that:

```text
persistent .grad
```

and:

```text
gradient currently being propagated
```

remain separate concepts.

This avoids incorrectly reusing gradients from an earlier backward traversal.

---

## Clearing Gradients

Gradients remain accumulated until they are explicitly cleared.

You can clear a Tensor directly:

```python
x.zero_grad()
```

or clear all model parameters using:

```python
model.zero_grad()
```

or:

```python
optimizer.zero_grad()
```

NanoNet resets gradients to:

```python
None
```

rather than allocating zero-filled arrays.

A typical training step therefore looks like:

```python
predictions = model(inputs)
loss = loss_fn(predictions, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Broadcasting

NumPy frequently performs operations between arrays with different shapes.

For example:

```text
batch: (32, 128)
bias:  (128,)
```

Adding them produces:

```text
(32, 128)
```

because NumPy broadcasts the bias across the batch dimension.

During backpropagation, however, the bias gradient must return to its original
shape:

```text
(128,)
```

NanoNet handles this using `unbroadcast()`.

Conceptually:

```text
Forward:

(32, 128) + (128,)
        ↓
    (32, 128)
```

During backward:

```text
upstream gradient
    (32, 128)
        │
        ▼
sum over broadcast axis
        │
        ▼
      (128,)
```

This is particularly important for Dense-layer biases.

---

## NumPy-Style Matrix Multiplication

NanoNet's `@` operator follows NumPy's `matmul` behavior.

It supports:

- vector @ vector
- matrix @ vector
- vector @ matrix
- matrix @ matrix
- batched matrix multiplication
- broadcasting across batch dimensions

Examples include:

```text
(3,) @ (3,)       → scalar

(4, 3) @ (3,)     → (4,)

(3,) @ (3, 5)     → (5,)

(4, 3) @ (3, 5)   → (4, 5)

(8, 4, 3) @ (3, 5)
                  → (8, 4, 5)
```

The backward pass handles NumPy's special treatment of one-dimensional inputs
by temporarily promoting them to matrix shapes.

After calculating the matrix derivatives, NanoNet:

1. removes temporary dimensions
2. reduces broadcast batch dimensions
3. restores gradients to each input's original shape

This allows Dense layers and more general matrix expressions to share the same
autograd implementation.

---

## Disabling Gradient Recording

Not every forward pass requires gradients.

During evaluation or inference, building a computational graph wastes work and
memory.

NanoNet provides:

```python
from nanonet_ml import no_grad
```

which can be used as:

```python
with no_grad():
    predictions = model(inputs)
```

Operations inside the context still compute their normal numerical results,
but they do not attach autograd `Function` nodes.

This is useful for:

- model evaluation
- prediction
- benchmark inference
- model summary shape inference
- other forward-only calculations

`model.evaluate(...)` automatically uses `no_grad()`.

---

## Graph Lifetime

NanoNet currently retains a graph while its output or intermediate tensors
remain referenced.

For example:

```python
y = model(x)
```

means `y` may still reference:

```text
y._grad_fn
y._parents
```

which indirectly retain earlier portions of the graph.

NanoNet therefore supports calling:

```python
y.backward()
```

more than once on the same graph.

Once the output and intermediate tensors are no longer referenced, normal
Python garbage collection can reclaim the graph.

NanoNet currently does not provide an explicit graph-freeing or
`retain_graph` API.

For a normal training loop this is usually not an issue because each batch
creates a fresh forward graph.

---

## Numerical Gradient Checking

Analytical gradients can be difficult to debug.

NanoNet therefore includes finite-difference gradient checking.

For a scalar function:

```text
f(x)
```

the numerical derivative can be approximated using the central-difference
formula:

```text
            f(x + ε) − f(x − ε)
f'(x) ≈    ───────────────────
                    2ε
```

NanoNet compares this numerical estimate with the gradient produced by
automatic differentiation.

Example:

```python
from nanonet_ml import Tensor
from nanonet_ml.gradcheck import gradcheck

a = Tensor(
    [1.5, -2.0],
    requires_grad=True,
)

b = Tensor(
    [0.5, 3.0],
    requires_grad=True,
)

result = gradcheck(
    lambda x, y: (x * y).sum(),
    [a, b],
)

print(result.passed)
```

Gradient checking is particularly useful for verifying:

- broadcasting
- matrix multiplication
- activation functions
- custom differentiable operations

Unused differentiable inputs are treated as having zero gradient.

---

## Summary

NanoNet's automatic-differentiation system works by:

```text
1. recording operations during the forward pass
2. connecting tensors into a computational graph
3. topologically ordering that graph
4. starting with the output gradient
5. applying local derivatives in reverse
6. combining gradients across multiple graph paths
7. reducing gradients created by broadcasting
8. accumulating final gradients in .grad
```

The implementation is intentionally small enough to inspect directly while
still supporting the operations required to train real neural networks.