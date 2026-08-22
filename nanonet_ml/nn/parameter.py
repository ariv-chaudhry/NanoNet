"""Trainable parameter tensors."""

from __future__ import annotations

from typing import Any

from nanonet_ml.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that represents a trainable model parameter.

    Parameters require gradients by default and are discovered automatically
    by ``Module.parameters()``.
    """

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"


def is_parameter(obj: Any) -> bool:
    return isinstance(obj, Parameter)
