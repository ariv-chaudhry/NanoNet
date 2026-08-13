# NanoNet Architecture

NanoNet is intentionally organized into small components so the path from a
tensor operation to a complete neural-network training loop remains easy to
follow.

The main architecture is:

```text
Tensor
  │
  ├── Automatic Differentiation
  │
Parameter
  │
Module
  │
  ├── Dense
  ├── ReLU
  ├── Dropout
  ├── Flatten
  │
Sequential
  │
Trainer
  │
Optimizer
```

Each part handles one major responsibility.

---

# 1. Tensor and Automatic Differentiation

The lowest level of NanoNet is:

```text
nanonet/tensor.py
```

A `Tensor` wraps a NumPy array and optionally participates in automatic
differentiation.

Example:

```python
from nanonet import Tensor

x = Tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True,
)
```

A Tensor stores:

```text
data
grad
requires_grad
_grad_fn
_parents
```

Conceptually:

```text
Tensor
├── numerical data
├── accumulated gradient
├── whether gradients are required
└── graph information
```

---

## Function Nodes

Differentiable operations are implemented using `Function` objects.

A Function knows how to:

```text
forward:
inputs → output

backward:
upstream gradient → input gradients
```

Examples include:

```text
Add
Sub
Mul
Div
Pow
MatMul
Sum
Mean
Reshape
Transpose
Exp
Log
Maximum
GetItem
```

When a differentiable operation runs, its output Tensor stores references to:

```text
_grad_fn
_parents
```

which creates the computational graph.

---

# 2. Gradient Recording

NanoNet normally records operations whenever at least one input requires
gradients.

For forward-only operations, graph construction can be disabled using:

```python
from nanonet import no_grad

with no_grad():
    predictions = model(inputs)
```

This is used internally during evaluation and summary shape inference.

It separates two concepts:

```text
training/evaluation mode
```

and:

```text
whether autograd graph recording is enabled
```

`model.eval()` controls layer behavior such as Dropout.

`no_grad()` controls graph construction.

---

# 3. Parameters

Trainable tensors use:

```text
nanonet/nn/parameter.py
```

A `Parameter` is a Tensor that is intended to be optimized.

For example, a Dense layer contains:

```text
weight
bias
```

as Parameters.

This allows NanoNet to distinguish between:

```text
ordinary intermediate tensors
```

and:

```text
trainable model state
```

---

# 4. Modules

The base neural-network component is:

```text
nanonet/nn/module.py
```

A `Module` automatically tracks:

```text
Parameter attributes
nested Module attributes
```

This allows recursive operations such as:

```python
model.parameters()
model.named_parameters()
model.zero_grad()
model.train()
model.eval()
model.state_dict()
model.num_parameters()
```

For example:

```text
Sequential
├── Dense
│   ├── weight
│   └── bias
├── ReLU
└── Dense
    ├── weight
    └── bias
```

Calling:

```python
model.parameters()
```

recursively collects the four trainable Parameters.

---

# 5. Layers

Standard layers live in:

```text
nanonet/layers/
```

Examples include:

```text
Dense
ReLU
Sigmoid
Tanh
Softmax
Dropout
Flatten
```

Layers are Modules.

They implement:

```python
forward(...)
```

and rely on Tensor operations for differentiation.

For example, a Dense layer performs:

```text
X @ W + b
```

The layer itself does not need a manually written Dense backward pass.

Instead:

```text
MatMul
```

and:

```text
Add
```

already know their derivatives.

This keeps the layer implementation small and makes the autograd engine
responsible for differentiation.

---

# 6. Sequential Models

`Sequential` stores Modules in order.

Example:

```python
from nanonet import Sequential
from nanonet.layers import Dense, ReLU

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
])
```

Conceptually:

```text
input
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
ReLU
  │
  ▼
Dense
  │
  ▼
output
```

Each layer receives the output of the previous layer.

---

# 7. Loss Functions

Loss functions live in:

```text
nanonet/losses/
```

NanoNet currently includes:

```text
MSELoss
CrossEntropyLoss
```

A loss converts model predictions and targets into a scalar Tensor.

That scalar becomes the starting point for:

```python
loss.backward()
```

---

## Cross Entropy

`CrossEntropyLoss` expects:

```text
logits:  (batch, classes)
targets: (batch,)
```

Targets must contain valid integer class labels.

Fractional values such as:

```text
1.5
```

are rejected instead of being silently converted to:

```text
1
```

The loss combines:

```text
log-softmax
+
negative log likelihood
```

using a numerically stable log-sum-exp calculation.

---

# 8. Optimizers

Optimizers live in:

```text
nanonet/optimizers/
```

Current implementations include:

```text
SGD
Adam
```

An optimizer stores references to model Parameters.

The training process is:

```text
loss.backward()
       │
       ▼
parameter.grad
       │
       ▼
optimizer.step()
       │
       ▼
parameter.data updated
```

NanoNet's Adam `weight_decay` uses **coupled L2 regularization**.

The L2 penalty is added to the gradient before Adam's moment calculations.

It is therefore not equivalent to AdamW's decoupled weight decay.

More details are available in:

```text
docs/optimizers.md
```

---

# 9. Data Loading

Data utilities live in:

```text
nanonet/data/
```

The package includes:

```text
Dataset
TensorDataset
DataLoader
MNIST loading utilities
```

`TensorDataset` wraps aligned arrays.

For example:

```python
from nanonet.data import TensorDataset

dataset = TensorDataset(
    X_train,
    y_train,
)
```

`DataLoader` handles:

```text
batching
shuffling
drop_last
deterministic seeding
```

The implementation is intentionally single-process to keep it readable.

---

# 10. MNIST Utilities

MNIST support is implemented in:

```text
nanonet/data/mnist.py
```

The loader can:

```text
download MNIST
cache the compressed IDX files
decode image and label files
flatten images
normalize pixels
```

The default cache location is:

```text
data/mnist/
```

The root `data/` directory is ignored by Git because downloaded datasets
should not be committed.

The Python package itself:

```text
nanonet/data/
```

remains tracked.

---

# 11. Training

Training orchestration lives in:

```text
nanonet/training/
```

The core class is:

```text
Trainer
```

A normal batch follows:

```text
forward
   ↓
loss
   ↓
zero_grad
   ↓
backward
   ↓
optimizer.step
```

In code:

```python
logits = model(X_batch)

loss = loss_fn(
    logits,
    y_batch,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Evaluation

Evaluation differs from training in two ways.

First:

```python
model.eval()
```

changes layer behavior such as Dropout.

Second:

```python
with no_grad():
```

disables computational-graph construction.

Conceptually:

```text
Training:
train mode + graph recording

Evaluation:
eval mode + no graph recording
```

---

# 12. Training History

Training metrics are stored in:

```text
nanonet/training/history.py
```

A `History` object can record values such as:

```text
loss
accuracy
validation loss
validation accuracy
```

These values can later be inspected or plotted.

---

# 13. Gradient Checking

Gradient verification is implemented in:

```text
nanonet/gradcheck.py
```

NanoNet compares:

```text
analytical gradient
```

from automatic differentiation with:

```text
numerical gradient
```

from central finite differences.

This helps detect incorrect backward implementations.

The gradient checker supports:

```text
broadcast operations
matrix operations
vector matmul
batched matmul
unused differentiable inputs
```

Unused differentiable inputs receive a numerical and analytical gradient of
zero.

---

# 14. Serialization

Model serialization lives in:

```text
nanonet/serialization.py
```

Parameters are saved using:

```text
NumPy .npz
```

with a JSON metadata sidecar.

For example:

```python
model.save(
    "models/mnist"
)
```

creates:

```text
models/mnist.npz
models/mnist.npz.meta.json
```

Both:

```python
model.save(
    "models/mnist"
)
```

and:

```python
model.save(
    "models/mnist.npz"
)
```

are supported.

Loading performs the same path normalization.

No Python pickle is required.

---

# 15. Metrics

Classification metrics live in:

```text
nanonet/metrics/
```

For classification, NanoNet calculates predictions using the largest logit:

```text
predicted class = argmax(logits)
```

Accuracy is then:

```text
correct predictions
───────────────────
 total predictions
```

---

# 16. Project Structure

The major repository directories are:

```text
NanoNet/
├── nanonet/
│   ├── tensor.py
│   ├── autograd.py
│   ├── gradcheck.py
│   ├── serialization.py
│   │
│   ├── nn/
│   ├── layers/
│   ├── losses/
│   ├── optimizers/
│   ├── data/
│   ├── metrics/
│   └── training/
│
├── examples/
├── benchmarks/
├── tests/
├── docs/
├── scripts/
├── README.md
├── LICENSE
└── pyproject.toml
```

---

# Design Principles

NanoNet follows several deliberate design goals.

## Correctness

Core mathematical operations are tested against known results and numerical
finite-difference gradients.

Important edge cases include:

```text
broadcasting
branching graphs
repeated backward
vector matrix multiplication
batched matrix multiplication
invalid class labels
serialization paths
```

---

## Readability

NanoNet prefers clear NumPy implementations over aggressive optimization.

The project is intended to make concepts such as:

```text
automatic differentiation
backpropagation
parameter registration
gradient accumulation
optimization
```

easy to inspect.

---

## Separation of Responsibilities

Each part of the framework has a distinct role:

```text
Tensor
    handles numerical operations and graph construction

Function
    defines local derivatives

Parameter
    represents trainable state

Module
    groups parameters and layers

Loss
    creates the scalar objective

Optimizer
    updates parameters

Trainer
    coordinates the training loop

DataLoader
    batches data
```

This separation keeps the implementation understandable while still allowing
the pieces to work together as a small neural-network framework.

---

## Educational Scope

NanoNet intentionally does not attempt to reproduce every feature of a
production framework.

It currently prioritizes:

```text
dense neural networks
autodiff correctness
training mechanics
readable implementation
```

over features such as:

```text
CUDA
distributed training
large model graphs
convolution kernels
production deployment
```

That limited scope is intentional.

---

# Licensing

NanoNet is distributed as **source-available software**.

Its source code may be viewed for educational reference and evaluation, but
the project is not released under an open-source license.

Permission is not granted to copy, modify, redistribute, republish,
sublicense, sell, or incorporate substantial portions of NanoNet into another
project without prior written permission from the copyright holder.

Copyright © 2026 Ariv Chaudhry. All rights reserved.

See the repository's [`LICENSE`](../LICENSE) file for the complete terms.