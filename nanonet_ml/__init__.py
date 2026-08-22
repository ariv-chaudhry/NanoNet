"""NanoNet: a lightweight educational neural-network framework built with NumPy.

Typical usage::

    import nanonet_ml as nn

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    y = model(nn.Tensor(...))
    model.inspect()
"""

from __future__ import annotations

from nanonet_ml._version import __version__
from nanonet_ml.layers import (
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
from nanonet_ml.losses import CrossEntropyLoss, MSELoss
from nanonet_ml.nn import Module, Parameter, Sequential
from nanonet_ml.optimizers import SGD, Adam, Optimizer
from nanonet_ml.tensor import Tensor, no_grad
from nanonet_ml.utils import manual_seed

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
