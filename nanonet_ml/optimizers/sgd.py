"""Stochastic Gradient Descent with optional momentum and weight decay."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from nanonet_ml.nn.parameter import Parameter
from nanonet_ml.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """SGD optimizer.

    Update rule with momentum and L2 weight decay::

        g_t = ∇L + weight_decay * θ
        v_t = momentum * v_{t-1} + g_t
        θ ← θ - lr * v_t

    Args:
        parameters: Model parameters.
        lr: Learning rate.
        momentum: Momentum factor in ``[0, 1)``.
        weight_decay: L2 penalty coefficient.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum}.")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocities: list[np.ndarray | None] = [None] * len(self.parameters)

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            if self.momentum != 0.0:
                v = self._velocities[i]
                if v is None:
                    v = np.zeros_like(param.data)
                v = self.momentum * v + grad
                self._velocities[i] = v
                update = v
            else:
                update = grad

            param.data = param.data - self.lr * update
