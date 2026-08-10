# Architecture

NanoNet is organized so each concern lives in a small, readable module.

```text
Tensor
  │
  ├── automatic differentiation
  │
Parameter
  │
Module
  │
  ├── Dense
  ├── ReLU
  ├── Dropout
  │
Sequential
  │
Trainer
  │
Optimizer
```

## Layers of the system

### Computation (`nanonet.tensor`, `nanonet.autograd`)

`Tensor` wraps a NumPy array and optionally builds a computational graph.
Each differentiable operator is a `Function` that knows how to:

1. compute a forward value
2. propagate gradients backward

Reverse-mode autodiff walks the graph in reverse topological order and
accumulates gradients with the chain rule. Broadcasting is handled by
`unbroadcast`, which reduces gradients back to each operand's original shape.

### Trainable modules (`nanonet.nn`, `nanonet.layers`)

`Module` tracks nested modules and `Parameter` objects. Calling
`model.parameters()` recursively collects every trainable weight and bias.

Layers such as `Dense`, `ReLU`, `Dropout`, and `Flatten` are Modules.
`Sequential` simply applies modules in order.

### Losses (`nanonet.losses`)

Loss modules map predictions and targets to a scalar Tensor. That scalar is
the root node for `backward()`.

`CrossEntropyLoss` fuses softmax and negative log-likelihood with a stable
log-sum-exp implementation so users pass **raw logits**.

### Optimization (`nanonet.optimizers`)

Optimizers own a list of Parameters. After `loss.backward()`, `optimizer.step()`
updates parameter `.data` using the accumulated `.grad` values.
`zero_grad()` sets `.grad` to `None` (not a zero array).

### Data loading (`nanonet.data`)

`Dataset` / `TensorDataset` provide sample access. `DataLoader` batches and
optionally shuffles indices. MNIST download/cache logic is isolated here so
the core framework never depends on network I/O at import time.

### Training orchestration (`nanonet.training`)

`Trainer` implements the epoch / batch loop:

```text
forward → loss → zero_grad → backward → step
```

`History` stores metrics for plotting. `Module.fit` / `Module.evaluate` are
thin wrappers around `Trainer`.

### Reliability helpers

* `nanonet.gradcheck` — finite-difference verification of analytical gradients
* `nanonet.serialization` — save/load parameters as `.npz` + JSON metadata
* `nanonet.metrics.accuracy` — classification accuracy from logits

## Design goals

1. **Correctness first** — broadcasting, matmul, and stable losses are tested.
2. **Readability** — prefer clear NumPy over clever micro-optimizations.
3. **Educational API** — familiar names (`Sequential`, `Adam`, `backward`) without
   hiding the mechanism behind another ML library.
