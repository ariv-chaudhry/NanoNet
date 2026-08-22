"""Adam optimizer implemented from first principles."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from nanonet_ml.nn.parameter import Parameter
from nanonet_ml.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba).

    Maintains exponential moving averages of the gradient (first moment) and
    squared gradient (second moment), with bias correction::

        m_t = β1 m_{t-1} + (1 - β1) g_t
        v_t = β2 v_{t-1} + (1 - β2) g_t²
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ ← θ - lr * m̂_t / (√v̂_t + ε)

    Args:
        parameters: Model parameters.
        lr: Learning rate.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        eps: Numerical stability term.
        weight_decay: L2 penalty added to gradients before Adam moment updates.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)

        if lr <= 0:
            raise ValueError(
                f"lr must be positive, got {lr}."
            )

        if not 0.0 <= beta1 < 1.0:
            raise ValueError(
                "beta1 must satisfy "
                f"0 <= beta1 < 1, got {beta1}."
            )

        if not 0.0 <= beta2 < 1.0:
            raise ValueError(
                "beta2 must satisfy "
                f"0 <= beta2 < 1, got {beta2}."
            )

        if eps <= 0:
            raise ValueError(
                f"eps must be positive, got {eps}."
            )

        if weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative, "
                f"got {weight_decay}."
            )

        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0

        self._m = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        self._v = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

    def step(self) -> None:
        self.t += 1

        b1_t = (
            1.0 - self.beta1**self.t
        )
        b2_t = (
            1.0 - self.beta2**self.t
        )

        for i, param in enumerate(
            self.parameters
        ):
            if param.grad is None:
                continue

            grad = param.grad

            if self.weight_decay != 0.0:
                grad = (
                    grad
                    + self.weight_decay * param.data
                )

            self._m[i] = (
                self.beta1 * self._m[i]
                + (1.0 - self.beta1) * grad
            )

            self._v[i] = (
                self.beta2 * self._v[i]
                + (1.0 - self.beta2)
                * (grad * grad)
            )

            m_hat = (
                self._m[i] / b1_t
            )
            v_hat = (
                self._v[i] / b2_t
            )

            param.data = (
                param.data
                - self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )