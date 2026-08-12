# NanoNet

A small neural-network framework built from scratch with Python and NumPy.

NanoNet started as a way for me to understand what libraries like PyTorch and TensorFlow are actually doing behind the scenes.

Instead of calling a high-level training API and letting a framework handle everything, I wanted to build the important pieces myself: tensors, computational graphs, automatic differentiation, backpropagation, optimizers, training loops, and model saving.

The goal isn't to compete with PyTorch. It's to make the mechanics of neural networks easier to see and understand.

NanoNet is small enough that you can follow the code from a single tensor operation all the way through backpropagation, but complete enough to train real models such as an MNIST digit classifier.

---

## How It Works

At a high level, training looks like this:

```text
Forward pass
────────────────────────────►

Input → Dense → ReLU → Dense → Loss

Gradient / backward pass
◄────────────────────────────
```

During the forward pass, NanoNet computes predictions while keeping track of the operations used to produce them.

Calling `backward()` then walks that graph in reverse and applies the chain rule to calculate gradients for every trainable parameter.

A typical MNIST network looks like:

```text
Input (784)
    │
    ▼
Dense (128)
    │
    ▼
ReLU
    │
    ▼
Dense (64)
    │
    ▼
ReLU
    │
    ▼
Dense (10)
    │
    ▼
Cross Entropy
```

---

## Features

NanoNet currently includes:

* **Tensors and reverse-mode automatic differentiation**
* NumPy-style **broadcasting** and matrix multiplication
* A lightweight **Module / Parameter system**
* `Sequential` models
* **Dense layers**
* ReLU, Sigmoid, Tanh, Softmax, Dropout, and Flatten
* Mean Squared Error and numerically stable Cross Entropy
* **SGD** with momentum and weight decay
* **Adam**
* Dataset and DataLoader utilities
* Training and validation loops
* Training history and plotting
* MNIST downloading and caching
* Model saving/loading using `.npz`
* Numerical gradient checking
* Model summaries and parameter counting

The core library does **not** use PyTorch, TensorFlow, JAX, autograd, or another automatic-differentiation package.

Just Python, NumPy, and a lot of chain rule.

---

## Why I Built This

Using modern machine-learning libraries makes it easy to build a neural network without really knowing what happens after calling:

```python
loss.backward()
optimizer.step()
```

I wanted to understand that part.

NanoNet implements the training process directly:

1. perform operations on tensors while recording a computational graph
2. work backward through that graph to calculate gradients
3. update parameters using an optimizer such as SGD or Adam
4. repeat that process across batches of training data

Building those pieces myself gave me a much better understanding of how backpropagation, gradient accumulation, broadcasting, weight initialization, and optimization actually work.

---

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install NanoNet:

```bash
pip install -e ".[dev]"
```

NumPy is the only core runtime dependency.

The development dependencies include pytest, coverage, Matplotlib, and Ruff.

---

## Quick Start

Here's a simple multilayer classifier:

```python
from nanonet import Sequential, manual_seed
from nanonet.layers import Dense, ReLU, Dropout
from nanonet.losses import CrossEntropyLoss
from nanonet.optimizers import Adam

manual_seed(42)

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dropout(0.2),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
])

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

model.fit(
    X_train,
    y_train,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
    batch_size=64,
)

accuracy = model.evaluate(X_test, y_test)
```

The API is intentionally familiar if you've used PyTorch or Keras before, but the implementation underneath it is NanoNet's own.

---

## Automatic Differentiation

The part of NanoNet I found most interesting to build was the automatic-differentiation engine.

For example:

```python
from nanonet import Tensor

x = Tensor(3.0, requires_grad=True)

y = x**2 + 2*x

y.backward()

print(x.grad)
# 8.0
```

Mathematically:

```text
y = x² + 2x

dy/dx = 2x + 2

At x = 3:

dy/dx = 8
```

NanoNet builds a graph from the operations used to create `y`, sorts the dependencies, and then walks that graph backward to calculate the derivative.

It also handles cases where a tensor contributes to the result through multiple paths:

```python
x = Tensor(2.0, requires_grad=True)

y = x*x + 3*x

y.backward()

print(x.grad)
# 7.0
```

Gradients from both branches are accumulated correctly.

---

## Building a Neural Network

Models can be created using `Sequential`:

```python
from nanonet import Sequential
from nanonet.layers import Dense, ReLU
from nanonet.losses import CrossEntropyLoss
from nanonet.optimizers import Adam

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
])

optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = CrossEntropyLoss()

model.summary(input_shape=(784,))

print("parameters:", model.num_parameters())
```

NanoNet automatically keeps track of trainable `Parameter` objects inside each layer.

Gradients can be cleared with either:

```python
optimizer.zero_grad()
```

or:

```python
model.zero_grad()
```

After clearing, each parameter's `.grad` is set back to `None` until the next backward pass.

---

## Training

You can train directly through `model.fit(...)`, or use the `Trainer` class if you want a little more control:

```python
from nanonet.training import Trainer

history = Trainer(model).fit(
    X_train,
    y_train,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
)
```

Training output looks something like:

```text
Epoch 1/10
loss: 0.4832 - accuracy: 85.72% - val_loss: 0.2314 - val_accuracy: 93.16%
```

Training history can also be plotted:

```python
history.plot(save_path="results/history.png")
```

This requires Matplotlib.

---

## MNIST

NanoNet includes an MNIST example using a fully connected network.

For a quick smoke test:

```bash
python examples/mnist_mlp.py --epochs 1 --train-limit 5000
```

During development, this reached **88.15% test accuracy after one epoch using 5,000 training samples**.

For a longer run:

```bash
python examples/mnist_mlp.py --epochs 10 --batch-size 64 --lr 0.001
```

Results will vary depending on initialization, configuration, and hardware.

More details about the model and training setup are in `docs/mnist.md`.

---

## Gradient Checking

Backpropagation bugs can be surprisingly hard to notice. A model can sometimes appear to train even when one of its gradients is slightly wrong.

NanoNet includes numerical gradient checking to help verify the autodiff engine:

```python
from nanonet import Tensor
from nanonet.gradcheck import gradcheck

a = Tensor([1.5, -2.0], requires_grad=True)
b = Tensor([0.5, 3.0], requires_grad=True)

result = gradcheck(
    lambda x, y: (x * y).sum(),
    [a, b]
)

print(result.passed)
print(result.max_abs_error)
print(result.max_rel_error)
```

The numerical derivative is estimated using finite differences and compared against NanoNet's calculated gradient.

This was especially useful for testing broadcasting, matrix multiplication, and activation functions.

---

## Architecture

NanoNet is split into a few main pieces:

| Component    | What it does                                                      |
| ------------ | ----------------------------------------------------------------- |
| `Tensor`     | Stores data and builds the computational graph                    |
| Autograd     | Calculates gradients using reverse-mode automatic differentiation |
| `Parameter`  | A Tensor that is meant to be trained                              |
| `Module`     | Base class for layers and models                                  |
| `Sequential` | Runs a collection of layers in order                              |
| Losses       | Turn predictions into a scalar training objective                 |
| Optimizers   | Update parameters using their gradients                           |
| DataLoader   | Handles batching and shuffling                                    |
| Trainer      | Runs the training and validation loops                            |

More detailed explanations are available in:

* `docs/architecture.md`
* `docs/autodiff.md`
* `docs/backpropagation.md`
* `docs/optimizers.md`

---

## Backpropagation

Suppose we have:

```text
y = (x * w) + b
```

The local derivatives are:

```text
∂y/∂x = w
∂y/∂w = x
∂y/∂b = 1
```

When the loss sends a gradient backward through this operation, those local derivatives are multiplied by the incoming gradient using the chain rule.

NanoNet repeats that process operation by operation until every trainable parameter has a gradient.

Conceptually:

```text
Forward pass
──────────────────────────────────►

Input → Dense → ReLU → Dense → Loss

        gradients flow backward

Input ← Dense ← ReLU ← Dense ← Loss

◄──────────────────────────────────
Backward pass
```

There's a longer walkthrough in `docs/backpropagation.md`.

---

## Benchmarks

NanoNet isn't designed to beat PyTorch.

PyTorch has highly optimized C/C++ kernels, optimized BLAS libraries, sophisticated memory management, GPU support, and years of engineering behind it.

NanoNet deliberately trades performance for readability.

The benchmark scripts make it possible to compare equivalent models:

```bash
python benchmarks/benchmark_nanonet.py
python benchmarks/benchmark_pytorch.py
python benchmarks/compare.py
```

The PyTorch benchmark is optional and requires PyTorch to be installed separately.

| Framework | Test Accuracy | Training Time |
| --------- | ------------- | ------------- |
| NanoNet   | run locally   | run locally   |
| PyTorch   | run locally   | run locally   |

I'd rather leave these values reproducible than put made-up benchmark numbers in the README.

---

## Project Structure

```text
nanonet/           # core framework
examples/          # autodiff, XOR, regression, and MNIST examples
benchmarks/        # NanoNet vs. optional PyTorch benchmarks
tests/             # pytest test suite
docs/              # architecture and math explanations
scripts/           # utility scripts
```

---

## Testing

Run the full test suite with:

```bash
pytest -v
```

For coverage:

```bash
pytest --cov=nanonet --cov-report=term-missing
```

And linting:

```bash
ruff check nanonet tests
```

You can also run the examples directly:

```bash
python examples/autodiff_demo.py
python examples/xor.py
python examples/regression.py
```

A lot of the core mathematical operations are tested against numerical finite-difference gradients rather than only checking output shapes.

---

## Limitations

NanoNet is intentionally fairly small.

Right now:

* it focuses on fully connected neural networks
* computation runs on NumPy / CPU only
* there are no convolutional layers yet
* there is no CUDA backend
* there is no multiprocessing DataLoader
* computational graphs don't support `retain_graph`
* performance is nowhere near a production ML framework

Those tradeoffs are intentional. The main goal is to keep the implementation understandable.

---

## Roadmap

Some things I'd like to experiment with in the future:

* Conv2D and MaxPool2D
* Batch normalization
* learning-rate schedulers
* more datasets and examples
* an optional GPU backend
* mixed-precision experiments

I'm trying to keep additions focused on things that are interesting to implement rather than turning NanoNet into a full PyTorch clone.

---

## License

NanoNet is **source-available**, not open source.

You are welcome to view the source code, study how the framework works, and
run an unmodified copy for personal, educational, research, or evaluation
purposes.

Modification, redistribution, incorporation into other projects, commercial
distribution, and representation of NanoNet or its source code as your own
work are not permitted without prior written permission.

Copyright © 2026 Ariv Chaudhry. All rights reserved.

See [`LICENSE`](LICENSE) for the complete terms.
