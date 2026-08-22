"""Shared helpers for model inspection and execution tracing."""

from __future__ import annotations

from typing import Any

import numpy as np

from nanonet_ml.inspection.report import ActivationStats
from nanonet_ml.nn.module import Module
from nanonet_ml.nn.parameter import Parameter
from nanonet_ml.tensor import Tensor


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


def activation_stats(value: Any) -> ActivationStats:
    """Compute activation statistics for a Tensor (metadata only, no copy retained)."""
    if not isinstance(value, Tensor):
        return ActivationStats(available=False)
    arr = np.asarray(value.data)
    size = int(arr.size)
    if size == 0:
        return ActivationStats(available=False, element_count=0)

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    nan_count = int(np.count_nonzero(nan_mask))
    inf_count = int(np.count_nonzero(inf_mask))
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return ActivationStats(
            available=True,
            nan_count=nan_count,
            inf_count=inf_count,
            element_count=size,
            zero_fraction=None,
            abs_max=None,
        )

    return ActivationStats(
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        zero_fraction=float(np.mean(arr == 0)),
        available=True,
        nan_count=nan_count,
        inf_count=inf_count,
        abs_max=float(np.max(np.abs(finite))),
        element_count=size,
    )


def gradient_norm(grad: Any) -> float | None:
    """L2 norm of a gradient array, or ``None`` if unavailable."""
    if grad is None:
        return None
    arr = np.asarray(grad, dtype=np.float64)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)] if not np.all(np.isfinite(arr)) else arr
    if finite.size == 0:
        return None
    return float(np.linalg.norm(finite))


def format_dtype(dtype: Any) -> str | None:
    """Concise user-facing dtype string (e.g. ``float64``)."""
    if dtype is None:
        return None
    text = str(dtype)
    if text.startswith("dtype(") and text.endswith(")"):
        text = text[6:-1].strip("'\"")
    if text.startswith("np."):
        text = text[3:]
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[8:-2].rsplit(".", 1)[-1]
    return text


def extract_tensor_metadata(value: Any) -> dict[str, Any]:
    """Extract JSON-safe tensor metadata without retaining arrays.

    Returns keys: ``shape``, ``dtype``, ``requires_grad``, ``has_grad``,
    ``grad_shape``. Missing fields are ``None`` / ``False`` as appropriate.
    """
    if not isinstance(value, Tensor):
        return {
            "shape": shape_of(value),
            "dtype": None,
            "requires_grad": None,
            "has_grad": False,
            "grad_shape": None,
        }
    has_grad = value.grad is not None
    return {
        "shape": tuple(value.shape),
        "dtype": format_dtype(value.dtype),
        "requires_grad": bool(value.requires_grad),
        "has_grad": has_grad,
        "grad_shape": tuple(value.grad.shape) if has_grad else None,
    }


def gradient_stats_rows(
    named_params: list[tuple[str, Parameter]],
) -> tuple[list[Any], bool]:
    """Build :class:`~nanonet_ml.inspection.GradientStats` rows for named parameters.

    Deduplicates shared parameters by object identity (first name wins).
    """
    from nanonet_ml.inspection.report import GradientStats

    rows: list[GradientStats] = []
    any_grad = False
    seen: set[int] = set()
    for name, param in named_params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if param.grad is None:
            rows.append(GradientStats(name=name, exists=False))
            continue
        any_grad = True
        g = np.asarray(param.grad, dtype=np.float64)
        finite = g[np.isfinite(g)] if g.size and not np.all(np.isfinite(g)) else g
        if finite.size == 0:
            rows.append(
                GradientStats(
                    name=name,
                    exists=True,
                    norm=None,
                    mean=None,
                    std=None,
                    min=None,
                    max=None,
                )
            )
            continue
        rows.append(
            GradientStats(
                name=name,
                exists=True,
                norm=float(np.linalg.norm(finite)),
                mean=float(np.mean(finite)),
                std=float(np.std(finite)),
                min=float(np.min(finite)),
                max=float(np.max(finite)),
            )
        )
    return rows, any_grad
