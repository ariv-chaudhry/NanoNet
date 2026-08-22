"""Flatten layer that preserves the batch dimension."""

from __future__ import annotations

from nanonet.nn.module import Module
from nanonet.tensor import Tensor


class Flatten(Module):
    """Flatten all dimensions except the batch dimension.

    Example::

        (batch, 28, 28) -> (batch, 784)
    """

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if x.ndim < 1:
            raise ValueError("Flatten expected input with at least 1 dimension.")
        batch = x.shape[0]
        return x.reshape(batch, -1)

    def __repr__(self) -> str:
        return "Flatten()"
