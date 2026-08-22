"""Activation function modules and functional helpers."""

from __future__ import annotations

import numpy as np

from nanonet_ml.nn.module import Module
from nanonet_ml.tensor import Function, Tensor


class ReLUFn(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a))
        return np.maximum(a, 0.0)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (a,) = self.saved_tensors
        return (grad_output * (a.data > 0),) if self.needs_input_grad[0] else (None,)


class SigmoidFn(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        # Stable sigmoid: for x >= 0 use 1/(1+e^{-x}); for x < 0 use e^x/(1+e^x).
        out = np.empty_like(a, dtype=np.float64)
        pos = a >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
        exp_x = np.exp(a[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        self.save_for_backward(Tensor(out))
        return out

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (out,) = self.saved_tensors
        s = out.data
        return (grad_output * s * (1.0 - s),) if self.needs_input_grad[0] else (None,)


class TanhFn(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        out = np.tanh(a)
        self.save_for_backward(Tensor(out))
        return out

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (out,) = self.saved_tensors
        return (grad_output * (1.0 - out.data**2),) if self.needs_input_grad[0] else (None,)


class SoftmaxFn(Function):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def _forward(self, a: np.ndarray) -> np.ndarray:
        # Numerically stable softmax: subtract max before exp.
        shifted = a - np.max(a, axis=self.axis, keepdims=True)
        exp_a = np.exp(shifted)
        out = exp_a / np.sum(exp_a, axis=self.axis, keepdims=True)
        self.save_for_backward(Tensor(out))
        return out

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        (out,) = self.saved_tensors
        s = out.data
        # Jacobian-vector product: s * (g - sum(g*s)) along the softmax axis.
        dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)
        return (s * (grad_output - dot),)


def relu(x: Tensor) -> Tensor:
    return ReLUFn().apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return SigmoidFn().apply(x)


def tanh(x: Tensor) -> Tensor:
    return TanhFn().apply(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return SoftmaxFn(axis=axis).apply(x)


class ReLU(Module):
    """Rectified Linear Unit: ``max(0, x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    """Sigmoid activation with a numerically stable implementation."""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Module):
    """Hyperbolic tangent activation."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Softmax(Module):
    """Softmax along ``axis`` (default last). Prefer CrossEntropyLoss on logits."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.axis)
