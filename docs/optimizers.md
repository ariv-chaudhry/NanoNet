# Optimizers

NanoNet implements optimizers manually in NumPy. They consume `Parameter`
gradients produced by autodiff and update `.data` in place.

## SGD

Stochastic gradient descent updates parameters using:

```text
θ ← θ − η ∇L
```

where:

- `θ` represents the model parameters
- `η` is the learning rate
- `∇L` is the gradient of the loss with respect to the parameters

### Weight Decay

NanoNet's SGD implementation can optionally apply L2 regularization through
`weight_decay`.

The gradient is adjusted before the parameter update:

```text
g ← ∇L + λθ
θ ← θ − ηg
```

where `λ` is the weight-decay coefficient.

This discourages parameters from growing unnecessarily large.

### Momentum

With momentum enabled, SGD maintains a velocity buffer:

```text
v ← μv + g
θ ← θ − ηv
```

where `μ` is the momentum coefficient.

Momentum smooths parameter updates over time. This can reduce oscillation in
directions where gradients change frequently while preserving movement in
directions where gradients consistently point the same way.

---

## Adam

Adam, short for **Adaptive Moment Estimation**, maintains two moving averages
for each parameter:

- **First moment (`m`)** — an exponential moving average of the gradients
- **Second moment (`v`)** — an exponential moving average of the squared gradients

If `weight_decay` is non-zero, NanoNet first applies an L2 penalty to the
gradient:

```text
g ← ∇L + λθ
```

The adjusted gradient is then used to update Adam's moment estimates:

```text
m_t = β₁m_{t-1} + (1 − β₁)g_t
v_t = β₂v_{t-1} + (1 − β₂)g_t²
```

Because both moving averages begin at zero, their initial values are biased
toward zero.

Adam corrects for this using **bias correction**:

```text
m̂_t = m_t / (1 − β₁ᵗ)
v̂_t = v_t / (1 − β₂ᵗ)
```

The parameter update is then:

```text
θ ← θ − η · m̂_t / (√v̂_t + ε)
```

where:

- `η` is the learning rate
- `β₁` controls the decay rate of the first moment
- `β₂` controls the decay rate of the second moment
- `ε` is a small constant used for numerical stability

### Intuition

The two moment estimates serve different purposes:

- `m̂` provides a smoothed estimate of the direction of the gradient.
- `√v̂` adapts the effective learning rate separately for each parameter.

Parameters that repeatedly receive large gradients therefore receive smaller
effective steps, while parameters with smaller gradient histories can receive
relatively larger steps.

### Default Hyperparameters

NanoNet uses the common Adam defaults:

```python
Adam(
    parameters,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
)
```

### Weight Decay in NanoNet's Adam

NanoNet implements `weight_decay` as **coupled L2 regularization**.

The penalty is added directly to the gradient:

```text
g ← ∇L + λθ
```

before Adam calculates its first and second moment estimates.

This is different from **AdamW**, which applies weight decay separately from
the gradient-based Adam update.

In other words:

```text
NanoNet Adam:
gradient → add L2 penalty → update moments → update parameters

AdamW:
gradient → update moments → Adam update
                         +
                  separate weight decay
```

NanoNet therefore should not describe its current `weight_decay` behavior as
"decoupled weight decay."

---

## Gradient Clearing

NanoNet accumulates gradients across calls to `backward()`.

For example:

```python
loss.backward()
```

adds new gradient contributions to each parameter's existing `.grad`.

Before processing the next training batch, gradients should normally be
cleared using either:

```python
optimizer.zero_grad()
```

or:

```python
model.zero_grad()
```

Both methods reset parameter gradients to:

```python
None
```

rather than allocating arrays filled with zeros.

A typical training step therefore follows this pattern:

```python
predictions = model(inputs)
loss = loss_fn(predictions, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Clearing gradients before each training step prevents gradients from previous
batches from being unintentionally accumulated into the next update.