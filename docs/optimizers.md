# Optimizers

NanoNet implements optimizers manually in NumPy. They consume Parameter
gradients produced by autodiff and update `.data` in place.

## SGD

Stochastic gradient descent:

```text
θ ← θ − η ∇L
```

With **weight decay** (L2 penalty on the gradient):

```text
g ← ∇L + λ θ
θ ← θ − η g
```

With **momentum**, a velocity buffer smooths updates:

```text
v ← μ v + g
θ ← θ − η v
```

Momentum helps traverse ravines and dampens noisy mini-batch gradients.

## Adam

Adam (Adaptive Moment Estimation) tracks two moving averages:

* **First moment** `m` — exponential average of gradients (mean direction)
* **Second moment** `v` — exponential average of squared gradients (scale)

```text
m_t = β₁ m_{t-1} + (1 − β₁) g_t
v_t = β₂ v_{t-1} + (1 − β₂) g_t²
```

Because moments start at zero, early estimates are biased low. Adam applies
**bias correction**:

```text
m̂_t = m_t / (1 − β₁ᵗ)
v̂_t = v_t / (1 − β₂ᵗ)
```

The parameter update is then:

```text
θ ← θ − η · m̂_t / (√v̂_t + ε)
```

Intuition:

* `m̂` points downhill with smoothed direction
* `√v̂` adapts the step size per coordinate (large gradient history → smaller steps)

Default hyperparameters in NanoNet match common practice:

```text
lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
```

## Gradient clearing

Both `optimizer.zero_grad()` and `model.zero_grad()` set each parameter's
`.grad` to `None`. Gradients **accumulate** across backward calls until cleared,
so training loops must zero gradients every step.
