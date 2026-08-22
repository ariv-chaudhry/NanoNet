"""Mean Squared Error loss."""

from __future__ import annotations

from nanonet.nn.module import Module
from nanonet.tensor import Tensor


class MSELoss(Module):
    """Mean squared error: ``mean((prediction - target)^2)``.

    Both prediction and target should be floating-point tensors of matching shape.
    """

    def forward(self, prediction: Tensor, target: Tensor | object) -> Tensor:
        if not isinstance(prediction, Tensor):
            prediction = Tensor(prediction)
        if not isinstance(target, Tensor):
            target = Tensor(target, requires_grad=False)
        if prediction.shape != target.shape:
            raise ValueError(
                f"MSELoss expected matching shapes, got prediction {prediction.shape} "
                f"and target {target.shape}."
            )
        diff = prediction - target
        return (diff * diff).mean()
