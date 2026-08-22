# Backpropagation

Backpropagation is the process NanoNet uses to calculate how much each
trainable parameter contributed to a model's final loss.

It is an application of the **chain rule** performed backward through the
computational graph.

---

## A Simple Example

Consider:

```text
y = xw + b
```

This computation can be separated into two operations:

```text
z = xw

y = z + b
```

The graph becomes:

```text
x ─────┐
       ▼
       × ───► z ───► + ───► y
       ▲               ▲
       │               │
       w               b
```

---

## Local Derivatives

Each operation only needs to know its own local derivative.

For:

```text
z = xw
```

the derivatives are:

```text
∂z/∂x = w

∂z/∂w = x
```

For:

```text
y = z + b
```

the derivatives are:

```text
∂y/∂z = 1

∂y/∂b = 1
```

---

## Applying the Chain Rule

Suppose a scalar loss `L` depends on `y`.

To calculate the gradient with respect to `x`:

```text
∂L/∂x
```

we combine the derivatives along the path:

```text
∂L/∂x = ∂L/∂y × ∂y/∂z × ∂z/∂x
```

Since:

```text
∂y/∂z = 1
```

and:

```text
∂z/∂x = w
```

we obtain:

```text
∂L/∂x = ∂L/∂y × w
```

Similarly:

```text
∂L/∂w = ∂L/∂y × x
```

and:

```text
∂L/∂b = ∂L/∂y
```

Every differentiable NanoNet operation implements this same idea.

---

## Forward Pass

During the forward pass:

```text
input → layer → activation → layer → loss
```

NanoNet performs two tasks.

First, it computes the numerical outputs.

Second, it records enough information to later differentiate those operations.

For example:

```python
predictions = model(inputs)
loss = loss_fn(predictions, targets)
```

creates the graph needed for:

```python
loss.backward()
```

---

## Backward Pass

The backward pass begins at the scalar loss.

Because:

```text
∂L/∂L = 1
```

the initial upstream gradient is:

```text
1
```

NanoNet then walks through the computational graph in reverse topological
order.

At each operation it:

1. receives the upstream gradient
2. calculates local derivatives
3. multiplies them using the chain rule
4. sends the resulting gradients to the operation's parents
5. accumulates gradients when multiple graph paths reach the same tensor

Conceptually:

```text
Forward:

Input → Dense → ReLU → Dense → Loss


Backward:

Input ← Dense ← ReLU ← Dense ← Loss
```

---

## Backpropagation Through a Dense Layer

A Dense layer computes:

```text
Y = XW + b
```

where:

```text
X = input batch
W = weight matrix
b = bias vector
```

Given an upstream gradient:

```text
G = ∂L/∂Y
```

the Dense-layer gradients are:

```text
∂L/∂X = G Wᵀ
```

```text
∂L/∂W = Xᵀ G
```

and:

```text
∂L/∂b = sum(G, axis=batch)
```

The bias gradient requires a sum because the same bias vector is broadcast
across every sample in the batch.

NanoNet does not hard-code Dense-layer backward logic inside the Dense layer.

Instead:

```text
X @ W
```

is handled by the autograd `MatMul` operation, while:

```text
+ b
```

is handled by the broadcast-aware `Add` operation.

Their gradients combine naturally through the computational graph.

---

## Backpropagation Through ReLU

ReLU is:

```text
ReLU(x) = max(0, x)
```

Its derivative is:

```text
1    if x > 0
0    if x < 0
```

Therefore an upstream gradient passes through positive activations and is
blocked for negative activations.

Conceptually:

```text
upstream gradient
        │
        ▼
    ReLU mask
        │
        ▼
gradient to input
```

---

## Branching Graphs

A tensor may affect a result through more than one path.

For example:

```python
from nanonet_ml import Tensor

x = Tensor(
    2.0,
    requires_grad=True,
)

y = x * x + 3 * x
```

Here `x` contributes through:

```text
x²
```

and:

```text
3x
```

so:

```text
dy/dx = 2x + 3
```

At:

```text
x = 2
```

we get:

```text
dy/dx = 7
```

NanoNet accumulates both gradient contributions into `x.grad`.

---

## Repeated Backward Calls

NanoNet also allows repeated backward calls on the same graph.

For example:

```python
from nanonet_ml import Tensor

x = Tensor(
    2.0,
    requires_grad=True,
)

y = x * x

y.backward()

print(x.grad)
```

produces:

```text
4.0
```

Calling:

```python
y.backward()
```

again produces:

```text
8.0
```

The important detail is that NanoNet does not reuse the already accumulated
gradient as the upstream gradient of the second traversal.

Each backward pass has an independent propagation buffer.

The persistent `.grad` field stores the accumulated result across calls.

---

## Parameter Updates

Backpropagation calculates gradients.

The optimizer uses those gradients to update the parameters.

For basic SGD:

```text
θ ← θ − η∇L
```

where:

- `θ` is a parameter
- `η` is the learning rate
- `∇L` is the gradient of the loss with respect to that parameter

For example:

```python
optimizer.zero_grad()

predictions = model(inputs)
loss = loss_fn(
    predictions,
    targets,
)

loss.backward()

optimizer.step()
```

The optimizer does not calculate the gradients itself.

That responsibility belongs to the autodiff engine.

---

## Training Loop

A standard NanoNet training step therefore follows:

```text
1. Forward pass
2. Calculate loss
3. Clear previous gradients
4. Backpropagate the new loss
5. Update the parameters
```

In code:

```python
predictions = model(inputs)
loss = loss_fn(
    predictions,
    targets,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This process repeats across mini-batches and epochs.

---

## Why Backpropagation Is Efficient

Suppose a model has thousands or millions of parameters.

Calculating the derivative of the loss separately with respect to every
parameter would be extremely expensive.

Reverse-mode autodiff calculates all those gradients during one reverse
traversal starting from the scalar loss.

That makes it particularly well suited to neural networks, where:

```text
many parameters → one scalar loss
```

is the standard training structure.

---

## Summary

Backpropagation in NanoNet can be summarized as:

```text
Forward:
inputs → operations → predictions → loss

Backward:
loss → local derivatives → accumulated parameter gradients

Optimization:
gradients → parameter updates
```

NanoNet separates these responsibilities:

```text
Tensor / Function
    ↓
automatic differentiation

Parameter
    ↓
stores trainable values and gradients

Optimizer
    ↓
uses gradients to update parameters
```

Keeping these responsibilities separate makes it easier to see how neural
network training actually works.