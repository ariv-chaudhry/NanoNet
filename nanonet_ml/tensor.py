"""Reverse-mode automatic differentiation primitives.

The Tensor class attaches lightweight Function nodes to each differentiable
operation. Calling ``backward()`` topologically sorts the graph and applies
the chain rule from the output toward the leaves.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np

from nanonet.utils import unbroadcast

if TYPE_CHECKING:
    from nanonet.inspection.report import ComputationGraph

_grad_enabled: ContextVar[bool] = ContextVar("nanonet_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    """Return whether operations are currently recording an autograd graph."""
    return _grad_enabled.get()


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable autograd graph construction.

    This is useful for inference, evaluation, and other forward-only work.
    The previous gradient-recording state is restored when the context exits,
    including when the context is nested or an exception is raised.
    """
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def _as_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


class Function:
    """Base class for differentiable operations.

    Subclasses store inputs needed for the backward pass and implement
    ``_forward`` / ``_backward``.
    """

    def __init__(self) -> None:
        self.saved_tensors: tuple[Tensor, ...] = ()
        self.needs_input_grad: list[bool] = []

    @property
    def op_name(self) -> str:
        """Human-readable operation label for graph inspection."""
        name = type(self).__name__
        if name.endswith("Function"):
            return name[: -len("Function")] or name
        if name.endswith("Fn"):
            return name[: -len("Fn")] or name
        return name

    def save_for_backward(self, *tensors: Tensor) -> None:
        self.saved_tensors = tensors

    def apply(self, *inputs: Any) -> Tensor:
        tensors = tuple(_as_tensor(x) for x in inputs)
        self.needs_input_grad = [t.requires_grad for t in tensors]
        raw = [t.data for t in tensors]
        result_data = self._forward(*raw)
        requires_grad = is_grad_enabled() and any(self.needs_input_grad)
        out = Tensor(result_data, requires_grad=requires_grad)
        if requires_grad:
            out._grad_fn = self
            out._parents = tensors
        return out

    def _forward(self, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        raise NotImplementedError


class Add(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._a_shape = a.shape
        self._b_shape = b.shape
        return a + b

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        ga = unbroadcast(grad_output, self._a_shape) if self.needs_input_grad[0] else None
        gb = unbroadcast(grad_output, self._b_shape) if self.needs_input_grad[1] else None
        return ga, gb


class Sub(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._a_shape = a.shape
        self._b_shape = b.shape
        return a - b

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        ga = unbroadcast(grad_output, self._a_shape) if self.needs_input_grad[0] else None
        gb = unbroadcast(-grad_output, self._b_shape) if self.needs_input_grad[1] else None
        return ga, gb


class Mul(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a), Tensor(b))
        self._a_shape = a.shape
        self._b_shape = b.shape
        return a * b

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.saved_tensors
        ga = unbroadcast(grad_output * b.data, self._a_shape) if self.needs_input_grad[0] else None
        gb = unbroadcast(grad_output * a.data, self._b_shape) if self.needs_input_grad[1] else None
        return ga, gb


class Div(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a), Tensor(b))
        self._a_shape = a.shape
        self._b_shape = b.shape
        return a / b

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.saved_tensors
        ga = unbroadcast(grad_output / b.data, self._a_shape) if self.needs_input_grad[0] else None
        gb = (
            unbroadcast(-grad_output * a.data / (b.data**2), self._b_shape)
            if self.needs_input_grad[1]
            else None
        )
        return ga, gb


class Pow(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a), Tensor(b))
        self._a_shape = a.shape
        self._b_shape = b.shape
        return a**b

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.saved_tensors
        out = a.data**b.data
        ga = None
        gb = None
        if self.needs_input_grad[0]:
            # d/da (a^b) = b * a^(b-1)
            ga = unbroadcast(
                grad_output * b.data * (a.data ** (b.data - 1.0)),
                self._a_shape,
            )
        if self.needs_input_grad[1]:
            # d/db (a^b) = a^b * log(a); define 0 for non-positive bases
            with np.errstate(divide="ignore", invalid="ignore"):
                log_a = np.where(a.data > 0, np.log(a.data), 0.0)
            gb = unbroadcast(grad_output * out * log_a, self._b_shape)
        return ga, gb


class Neg(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        return (-grad_output,) if self.needs_input_grad[0] else (None,)


class MatMul(Function):
    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a), Tensor(b))
        self._a_shape = a.shape
        self._b_shape = b.shape
        return np.matmul(a, b)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.saved_tensors
        a_data = a.data
        b_data = b.data

        # np.matmul treats 1-D operands specially: a leading or trailing
        # dimension is temporarily inserted for the matrix product and removed
        # from the result. Recreate those promoted shapes for the gradient math.
        a_was_1d = a_data.ndim == 1
        b_was_1d = b_data.ndim == 1
        a_mat = a_data[np.newaxis, :] if a_was_1d else a_data
        b_mat = b_data[:, np.newaxis] if b_was_1d else b_data

        grad_mat = np.asarray(grad_output)
        if a_was_1d and b_was_1d:
            grad_mat = grad_mat.reshape(1, 1)
        elif a_was_1d:
            grad_mat = np.expand_dims(grad_mat, axis=-2)
        elif b_was_1d:
            grad_mat = np.expand_dims(grad_mat, axis=-1)

        ga = None
        gb = None
        if self.needs_input_grad[0]:
            ga_mat = np.matmul(grad_mat, np.swapaxes(b_mat, -1, -2))
            ga = unbroadcast(ga_mat, a_mat.shape).reshape(self._a_shape)
        if self.needs_input_grad[1]:
            gb_mat = np.matmul(np.swapaxes(a_mat, -1, -2), grad_mat)
            gb = unbroadcast(gb_mat, b_mat.shape).reshape(self._b_shape)
        return ga, gb


class Sum(Function):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, a: np.ndarray) -> np.ndarray:
        self._input_shape = a.shape
        return np.sum(a, axis=self.axis, keepdims=self.keepdims)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        grad = grad_output
        if self.axis is not None and not self.keepdims:
            axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            # Expand reduced axes so broadcasting restores the input shape.
            shape = list(self._input_shape)
            for ax in sorted(ax % len(shape) for ax in axis):
                shape[ax] = 1
            grad = np.asarray(grad).reshape(shape)
        return (np.broadcast_to(grad, self._input_shape).copy(),)


class Mean(Function):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, a: np.ndarray) -> np.ndarray:
        self._input_shape = a.shape
        if self.axis is None:
            self._n = a.size
        else:
            axes = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            self._n = int(np.prod([a.shape[ax] for ax in axes]))
        return np.mean(a, axis=self.axis, keepdims=self.keepdims)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        grad = grad_output / self._n
        if self.axis is not None and not self.keepdims:
            axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            shape = list(self._input_shape)
            for ax in sorted(ax % len(shape) for ax in axis):
                shape[ax] = 1
            grad = np.asarray(grad).reshape(shape)
        return (np.broadcast_to(grad, self._input_shape).copy(),)


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def _forward(self, a: np.ndarray) -> np.ndarray:
        self._input_shape = a.shape
        return a.reshape(self.shape)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output.reshape(self._input_shape),)


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.axes = axes

    def _forward(self, a: np.ndarray) -> np.ndarray:
        self._input_ndim = a.ndim
        return np.transpose(a, axes=self.axes)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        if self.axes is None:
            return (np.transpose(grad_output),)
        # Invert the permutation applied in the forward pass.
        inv = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv[ax] = i
        return (np.transpose(grad_output, axes=tuple(inv)),)


class Exp(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        self.save_for_backward(Tensor(out))
        return out

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (out,) = self.saved_tensors
        return (grad_output * out.data,) if self.needs_input_grad[0] else (None,)


class Log(Function):
    def _forward(self, a: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a))
        return np.log(a)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (a,) = self.saved_tensors
        return (grad_output / a.data,) if self.needs_input_grad[0] else (None,)


class Maximum(Function):
    """Element-wise maximum; used for ReLU (maximum(x, 0))."""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.save_for_backward(Tensor(a), Tensor(b))
        self._a_shape = a.shape
        self._b_shape = b.shape
        return np.maximum(a, b)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.saved_tensors
        # Split ties evenly so gradients remain well-defined.
        equal = a.data == b.data
        mask_a = np.where(
            equal,
            0.5,
            (a.data > b.data).astype(np.float64),
        )
        mask_b = np.where(
            equal,
            0.5,
            (b.data > a.data).astype(np.float64),
        )
        ga = (
            unbroadcast(grad_output * mask_a, self._a_shape)
            if self.needs_input_grad[0]
            else None
        )
        gb = (
            unbroadcast(grad_output * mask_b, self._b_shape)
            if self.needs_input_grad[1]
            else None
        )
        return ga, gb


class GetItem(Function):
    def __init__(self, index: Any) -> None:
        super().__init__()
        self.index = index

    def _forward(self, a: np.ndarray) -> np.ndarray:
        self._input_shape = a.shape
        return a[self.index]

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        grad = np.zeros(self._input_shape, dtype=grad_output.dtype)
        np.add.at(grad, self.index, grad_output)
        return (grad,)


class Tensor:
    """NumPy array wrapper with reverse-mode automatic differentiation.

    Attributes:
        data: Underlying NumPy ndarray (float64 by default).
        grad: Accumulated gradient, or ``None`` if not yet computed / zeroed.
        requires_grad: Whether this tensor participates in autodiff.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        *,
        dtype: np.dtype | type | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            arr = data.data
        else:
            arr = np.asarray(
                data,
                dtype=dtype if dtype is not None else np.float64,
            )
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        elif arr.dtype.kind not in "fc":
            # Promote integers to float so gradients stay meaningful.
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = bool(requires_grad)
        self._grad_fn: Function | None = None
        self._parents: tuple[Tensor, ...] = ()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.data.copy()

    def item(self) -> float:
        """Return the Python scalar for a 0-d / single-element tensor."""
        return float(self.data.item())

    def zero_grad(self) -> None:
        """Clear accumulated gradients by setting ``grad`` to ``None``.

        NanoNet uses ``None`` (rather than a zero array) to indicate that no
        gradient has been accumulated since the last reset. Optimizers treat
        ``None`` as "skip this parameter".
        """
        self.grad = None

    def detach(self) -> Tensor:
        """Return an independent copy of this tensor detached from the graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def graph(self, *, verbose: bool = True) -> ComputationGraph:
        """Inspect the autograd computation graph ending at this tensor.

        Traverses ``_parents`` / ``_grad_fn`` from this tensor (the root) and
        builds a structured :class:`~nanonet.inspection.ComputationGraph`.
        Does not call ``backward()``, clear gradients, or mutate the graph.

        Gradient presence (``has_grad``) reflects whatever is already stored
        on tensors — call ``backward()`` first if you want post-backward
        gradient metadata.

        Args:
            verbose: If True (default), print the formatted graph to stdout.
                If False, return the graph without printing.

        Returns:
            A :class:`~nanonet.inspection.ComputationGraph`.
        """
        from nanonet.inspection.graph import inspect_computation_graph

        return inspect_computation_graph(self, verbose=verbose)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, gradient: np.ndarray | Tensor | None = None) -> None:
        """Compute gradients of this tensor w.r.t. all reachable leaves.

        For a non-scalar output, ``gradient`` must be provided and match
        ``self.shape``. Gradients accumulate in each tensor's ``.grad`` field
        across backward calls until ``zero_grad()`` is used. The graph itself
        is retained, so the same result may be backpropagated more than once.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward on a Tensor that does not require grad."
            )

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() requires a gradient argument for non-scalar outputs "
                    f"(got shape {self.shape})."
                )
            grad = np.ones_like(self.data)
        elif isinstance(gradient, Tensor):
            grad = np.asarray(gradient.data, dtype=self.data.dtype)
        else:
            grad = np.asarray(gradient, dtype=self.data.dtype)

        if grad.shape != self.data.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match tensor shape {self.shape}."
            )

        # Build a topological order of the differentiable graph.
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node._parents:
                if parent.requires_grad:
                    build(parent)
            topo.append(node)

        build(self)

        # Use a per-call gradient buffer for propagation. Tensor.grad stores the
        # persistent accumulated gradient, but historical gradients must never be
        # propagated through the graph again on a later backward() call.
        pending: dict[int, np.ndarray] = {id(self): grad}

        for node in reversed(topo):
            node_grad = pending.get(id(node))
            if node_grad is None:
                continue

            node.grad = (
                node_grad
                if node.grad is None
                else node.grad + node_grad
            )

            if node._grad_fn is None:
                continue

            grads = node._grad_fn._backward(node_grad)
            for parent, parent_grad in zip(node._parents, grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                parent_grad = np.asarray(
                    parent_grad,
                    dtype=parent.data.dtype,
                )
                parent_id = id(parent)
                if parent_id in pending:
                    pending[parent_id] = (
                        pending[parent_id] + parent_grad
                    )
                else:
                    pending[parent_id] = parent_grad

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> Tensor:
        return Add().apply(self, other)

    def __radd__(self, other: Any) -> Tensor:
        return Add().apply(other, self)

    def __sub__(self, other: Any) -> Tensor:
        return Sub().apply(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        return Sub().apply(other, self)

    def __mul__(self, other: Any) -> Tensor:
        return Mul().apply(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return Mul().apply(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        return Div().apply(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        return Div().apply(other, self)

    def __pow__(self, other: Any) -> Tensor:
        return Pow().apply(self, other)

    def __rpow__(self, other: Any) -> Tensor:
        return Pow().apply(other, self)

    def __matmul__(self, other: Any) -> Tensor:
        return MatMul().apply(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        return MatMul().apply(other, self)

    def __neg__(self) -> Tensor:
        return Neg().apply(self)

    def __getitem__(self, index: Any) -> Tensor:
        return GetItem(index).apply(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Tensor methods
    # ------------------------------------------------------------------
    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return Sum(axis=axis, keepdims=keepdims).apply(self)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return Mean(axis=axis, keepdims=keepdims).apply(self)

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            new_shape = shape[0]
        else:
            new_shape = shape
        return Reshape(tuple(int(s) for s in new_shape)).apply(self)

    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            return Transpose(None).apply(self)
        if len(axes) == 1 and isinstance(axes[0], tuple):
            return Transpose(axes[0]).apply(self)
        return Transpose(tuple(axes)).apply(self)

    def exp(self) -> Tensor:
        return Exp().apply(self)

    def log(self) -> Tensor:
        return Log().apply(self)

    def maximum(self, other: Any) -> Tensor:
        return Maximum().apply(self, other)

    @staticmethod
    def maximum_of(a: Any, b: Any) -> Tensor:
        return Maximum().apply(a, b)


# Public function aliases operating on Tensors / arrays
def exp(x: Any) -> Tensor:
    return _as_tensor(x).exp()


def log(x: Any) -> Tensor:
    return _as_tensor(x).log()


def maximum(a: Any, b: Any) -> Tensor:
    return Tensor.maximum_of(a, b)