"""NanoNet: a lightweight educational neural-network framework built with NumPy.

Typical usage::

    import nanonet as nn

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    y = model(nn.Tensor(...))
    model.inspect()
"""

from __future__ import annotations

from nanonet._version import __version__
from nanonet.layers import (
    Dense,
    Dropout,
    Flatten,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from nanonet.losses import CrossEntropyLoss, MSELoss
from nanonet.nn import Module, Parameter, Sequential
from nanonet.optimizers import SGD, Adam, Optimizer
from nanonet.tensor import Tensor, no_grad
from nanonet.utils import manual_seed

__all__ = [
    "__version__",
    # Core
    "Tensor",
    "no_grad",
    "manual_seed",
    # Modules
    "Module",
    "Parameter",
    "Sequential",
    # Layers
    "Dense",
    "Linear",
    "Dropout",
    "Flatten",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    # Optimizers
    "Optimizer",
    "SGD",
    "Adam",
    # Losses
    "MSELoss",
    "CrossEntropyLoss",
]
