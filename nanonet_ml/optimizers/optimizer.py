"""Optimizer base class."""

from __future__ import annotations

from collections.abc import Iterable

from nanonet_ml.nn.parameter import Parameter


class Optimizer:
    """Base class for parameter optimizers.

    Args:
        parameters: Iterable of Parameters to update.
    """

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self.parameters: list[Parameter] = list(parameters)
        if not self.parameters:
            raise ValueError("Optimizer received an empty parameter list.")

    def step(self) -> None:
        """Apply one optimization step using accumulated gradients."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Clear gradients on all managed parameters (sets ``grad`` to ``None``)."""
        for param in self.parameters:
            param.zero_grad()
