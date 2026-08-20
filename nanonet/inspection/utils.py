"""Shared helpers for model inspection and execution tracing."""

from __future__ import annotations

from typing import Any

import numpy as np

from nanonet.nn.module import Module
from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor


def iter_named_modules(
    module: Module,
    prefix: str = "",
    *,
    include_root: bool = True,
) -> list[tuple[str, Module]]:
    """Return ``(hierarchical_name, module)`` for ``module`` and descendants."""
    result: list[tuple[str, Module]] = []
    if include_root:
        result.append((prefix, module))
    for name, child in module._modules.items():
        full = f"{prefix}.{name}" if prefix else name
        result.extend(iter_named_modules(child, full, include_root=True))
    return result


def leaf_modules(module: Module) -> list[tuple[str, Module]]:
    """Return modules with no registered children, with hierarchical names.

    If ``module`` itself has no children (e.g. a bare ``Dense``), it is returned
    as a single leaf named after its class.
    """
    if not module._modules:
        return [(type(module).__name__, module)]

    leaves: list[tuple[str, Module]] = []
    for name, child in module._modules.items():
        leaves.extend(_leaf_modules_from(child, name))
    return leaves


def _leaf_modules_from(module: Module, prefix: str) -> list[tuple[str, Module]]:
    if not module._modules:
        return [(prefix, module)]
    leaves: list[tuple[str, Module]] = []
    for name, child in module._modules.items():
        full = f"{prefix}.{name}" if prefix else name
        leaves.extend(_leaf_modules_from(child, full))
    return leaves


def count_parameters(module: Module, *, recursive: bool = True) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter element counts.

    Args:
        module: Module to count.
        recursive: If True, include nested modules. If False, only parameters
            registered directly on ``module`` (``module._parameters``).
    """
    total = 0
    trainable = 0
    seen: set[int] = set()
    params: list[Parameter]
    if recursive:
        params = module.parameters()
    else:
        params = list(module._parameters.values())
    for param in params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        n = int(param.size)
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def parameter_memory_bytes(module: Module) -> int:
    """Sum ``nbytes`` of unique parameters under ``module``."""
    total = 0
    seen: set[int] = set()
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.data.nbytes)
    return total


def shape_of(value: Any) -> tuple[int, ...] | None:
    """Best-effort shape extraction for Tensors / ndarrays."""
    if isinstance(value, Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    return None


def collect_tensors(value: Any) -> list[Tensor]:
    """Flatten Tensor objects from a value, tuple, or list (one level deep)."""
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        found: list[Tensor] = []
        for item in value:
            found.extend(collect_tensors(item))
        return found
    return []


def first_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    if kwargs:
        return next(iter(kwargs.values()))
    return None


def format_duration(seconds: float) -> str:
    """Format a duration in seconds using ASCII-safe units."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.4f} s"
