"""Numerical gradient checking utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nanonet.tensor import Tensor


@dataclass
class GradCheckResult:
    """Diagnostics from a finite-difference gradient check."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    analytical: np.ndarray
    numerical: np.ndarray

    def __bool__(self) -> bool:
        return self.passed


def _rel_error(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    denom = np.maximum(
        np.abs(a) + np.abs(b),
        1e-12,
    )
    return np.abs(a - b) / denom


def numerical_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical gradient of a scalar function."""
    x = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    grad = np.zeros_like(x)

    it = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + epsilon
        f_pos = func(x)

        x[idx] = original - epsilon
        f_neg = func(x)

        x[idx] = original

        grad[idx] = (
            f_pos - f_neg
        ) / (2.0 * epsilon)

        it.iternext()

    return grad


def gradcheck(
    func: Callable[..., Tensor],
    inputs: list[Tensor] | tuple[Tensor, ...],
    *,
    epsilon: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradCheckResult:
    """Compare analytical gradients from NanoNet against finite differences.

    ``func(*inputs)`` must return a scalar Tensor.

    Args:
        func: Differentiable function mapping Tensors to a scalar Tensor.
        inputs: Tensors with ``requires_grad=True`` to check.
        epsilon: Finite-difference step.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        ``GradCheckResult`` with pass/fail and error statistics.
    """
    inputs = list(inputs)

    for t in inputs:
        if not t.requires_grad:
            raise ValueError(
                "All inputs to gradcheck must have requires_grad=True."
            )

    # Analytical gradients
    for t in inputs:
        t.zero_grad()

    out = func(*inputs)

    if out.data.size != 1:
        raise ValueError(
            f"gradcheck requires a scalar output, got shape {out.shape}."
        )

    out.backward()

    analytical_parts = [
        (
            t.grad
            if t.grad is not None
            else np.zeros_like(t.data)
        ).reshape(-1)
        for t in inputs
    ]

    analytical = np.concatenate(
        analytical_parts
    )

    # Numerical gradients
    numerical_parts: list[np.ndarray] = []

    for i, t in enumerate(inputs):
        base = t.data.copy()

        def scalar_fn(
            perturbed: np.ndarray,
            index: int = i,
            original: np.ndarray = base,
        ) -> float:
            restored = []

            for j, inp in enumerate(inputs):
                if j == index:
                    restored.append(
                        Tensor(
                            perturbed,
                            requires_grad=True,
                        )
                    )
                else:
                    restored.append(
                        Tensor(
                            inp.data.copy(),
                            requires_grad=True,
                        )
                    )

            return float(
                func(*restored).data
            )

        numerical_parts.append(
            numerical_gradient(
                scalar_fn,
                base,
                epsilon=epsilon,
            ).reshape(-1)
        )

    numerical = np.concatenate(
        numerical_parts
    )

    abs_err = np.abs(
        analytical - numerical
    )

    rel_err = _rel_error(
        analytical,
        numerical,
    )

    max_abs = (
        float(np.max(abs_err))
        if abs_err.size
        else 0.0
    )

    max_rel = (
        float(np.max(rel_err))
        if rel_err.size
        else 0.0
    )

    passed = bool(
        np.allclose(
            analytical,
            numerical,
            rtol=rtol,
            atol=atol,
        )
    )

    return GradCheckResult(
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        analytical=analytical,
        numerical=numerical,
    )