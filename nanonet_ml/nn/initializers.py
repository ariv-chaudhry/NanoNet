"""Weight initialization helpers for Dense layers."""

from __future__ import annotations

import math

import numpy as np

from nanonet_ml.utils import get_rng


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    """Return an array of zeros."""
    return np.zeros(shape, dtype=np.float64)


def ones(shape: tuple[int, ...]) -> np.ndarray:
    """Return an array of ones."""
    return np.ones(shape, dtype=np.float64)


def normal(shape: tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Sample from a normal distribution."""
    return get_rng().normal(mean, std, size=shape).astype(np.float64)


def uniform(shape: tuple[int, ...], low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Sample from a uniform distribution."""
    return get_rng().uniform(low, high, size=shape).astype(np.float64)


def xavier_uniform(shape: tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """Xavier/Glorot uniform initialization.

    Designed for layers followed by sigmoid/tanh. Variance is scaled by
    ``fan_in + fan_out`` so activations neither explode nor vanish early in
    training.
    """
    fan_in, fan_out = _fans(shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, -limit, limit)


def xavier_normal(shape: tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """Xavier/Glorot normal initialization."""
    fan_in, fan_out = _fans(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, 0.0, std)


def kaiming_uniform(shape: tuple[int, ...], a: float = 0.0) -> np.ndarray:
    """Kaiming/He uniform initialization (good default for ReLU networks)."""
    fan_in, _ = _fans(shape)
    gain = math.sqrt(2.0 / (1.0 + a**2))
    limit = gain * math.sqrt(3.0 / fan_in)
    return uniform(shape, -limit, limit)


def kaiming_normal(shape: tuple[int, ...], a: float = 0.0) -> np.ndarray:
    """Kaiming/He normal initialization (good default for ReLU networks)."""
    fan_in, _ = _fans(shape)
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    return normal(shape, 0.0, std)


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        fan_in = fan_out = shape[0] if shape else 1
    else:
        # Dense weights are stored as (in_features, out_features).
        fan_in, fan_out = int(shape[0]), int(shape[1])
    return fan_in, fan_out
