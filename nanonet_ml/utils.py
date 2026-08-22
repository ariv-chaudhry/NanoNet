"""Shared utilities for NanoNet."""

from __future__ import annotations

import numpy as np

# Global NumPy Generator used for reproducible initialization, dropout, and shuffling.
_rng = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Seed NanoNet's NumPy random generator for reproducible runs.

    Affects weight initialization, Dropout masks, and DataLoader shuffling.
    Perfect cross-platform bit-for-bit determinism is not guaranteed, but
    repeated runs in the same environment with the same seed behave predictably.

    Args:
        seed: Non-negative integer seed.
    """
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Return the shared NumPy Generator used by NanoNet."""
    return _rng


def unbroadcast(gradient: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to ``target_shape``.

    When an operation broadcasts operands (e.g. ``(32, 128) + (128,)``), the
    upstream gradient has the broadcasted shape. Gradients for the smaller
    operand must be summed over the dimensions that were expanded.

    Args:
        gradient: Upstream gradient with broadcasted shape.
        target_shape: Original shape of the operand that was broadcast.

    Returns:
        Gradient reshaped/reduced to ``target_shape``.
    """
    grad = np.asarray(gradient)

    if target_shape == ():
        return np.asarray(np.sum(grad))

    # Sum out leading dimensions that were added by broadcasting.
    ndim_added = grad.ndim - len(target_shape)
    for _ in range(ndim_added):
        grad = grad.sum(axis=0)

    # Sum over axes that were size-1 in the original operand.
    reduce_axes = tuple(
        i for i, (g_size, t_size) in enumerate(zip(grad.shape, target_shape)) if t_size == 1 and g_size > 1
    )
    if reduce_axes:
        grad = grad.sum(axis=reduce_axes, keepdims=True)

    return grad.reshape(target_shape)
