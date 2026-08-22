"""Dropout regularization layer."""

from __future__ import annotations

import numpy as np

from nanonet_ml.nn.module import Module
from nanonet_ml.tensor import Function, Tensor
from nanonet_ml.utils import get_rng


class DropoutFn(Function):
    def __init__(self, mask: np.ndarray) -> None:
        super().__init__()
        self.mask = mask

    def _forward(self, a: np.ndarray) -> np.ndarray:
        return a * self.mask

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        return (grad_output * self.mask,) if self.needs_input_grad[0] else (None,)


class Dropout(Module):
    """Inverted dropout.

    During training, zeros activations with probability ``p`` and scales the
    remainder by ``1 / (1 - p)`` so expected values match evaluation.

    During evaluation, returns the input unchanged.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must satisfy 0 <= p < 1, got {p}.")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        mask = (get_rng().random(x.shape) < keep_prob).astype(np.float64) / keep_prob
        return DropoutFn(mask).apply(x)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
