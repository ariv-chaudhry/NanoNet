"""Fully-connected (dense / linear) layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nanonet.nn.initializers import kaiming_uniform, zeros
from nanonet.nn.module import Module
from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor


class Dense(Module):
    """Affine transformation: ``y = x @ W + b``.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, learn an additive bias of shape ``(out_features,)``.
        weight_init: Callable ``(shape) -> ndarray`` for weight initialization.
            Defaults to Kaiming uniform (good for ReLU networks).
        bias_init: Callable for bias initialization (default zeros).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: Callable[[tuple[int, ...]], np.ndarray] | None = None,
        bias_init: Callable[[tuple[int, ...]], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        init_w = weight_init or kaiming_uniform
        self.weight = Parameter(init_w((self.in_features, self.out_features)))

        if bias:
            init_b = bias_init or zeros
            self.bias: Parameter | None = Parameter(init_b((self.out_features,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if x.ndim < 2:
            raise ValueError(
                f"{type(self).__name__} expected input with at least 2 dimensions "
                f"(batch, features), got shape {x.shape}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{type(self).__name__} expected last input dimension {self.in_features} "
                f"but received {x.shape[-1]}."
            )

        # Support (batch, features) and higher-rank inputs by flattening leading dims.
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)

        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], self.out_features)
        return out

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class Linear(Dense):
    """Fully-connected linear layer.

    Identical to :class:`Dense`. Provided as a familiar public alias for
    ``import nanonet as nn`` / ``nn.Linear(...)`` usage.
    """

    pass
