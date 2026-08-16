"""Shared helpers for NanoNet vs PyTorch empirical evaluation."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path("results")


@dataclass
class TimingStats:
    """Summary statistics over repeated wall-clock measurements."""

    mean: float
    std: float
    min: float
    max: float
    runs: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch RNGs."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Torch may be absent, or the platform may lack a deterministic path.
        pass


def require_torch():
    """Import torch or exit with an install hint."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for this evaluation script.\n"
            "Install with: pip install 'nanonet[benchmark]'"
        ) from exc
    return torch


def environment_metadata(
    *,
    dtype: str,
    device: str = "CPU",
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect lightweight, cross-platform environment metadata."""
    import nanonet

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "nanonet_version": getattr(nanonet, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "os": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "dtype": dtype,
        "device": device,
    }
    if seed is not None:
        meta["seed"] = seed
    try:
        import torch

        meta["pytorch_version"] = torch.__version__
    except ImportError:
        meta["pytorch_version"] = None
    if extra:
        meta.update(extra)
    return meta


def compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    name: str = "array",
) -> dict[str, Any]:
    """Compare two arrays and return absolute/relative error diagnostics."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    abs_err = np.abs(a - b)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    rel_err = abs_err / denom
    max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
    mean_abs = float(np.mean(abs_err)) if abs_err.size else 0.0
    max_rel = float(np.max(rel_err)) if rel_err.size else 0.0
    passed = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    return {
        "name": name,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "allclose": passed,
        "rtol": rtol,
        "atol": atol,
    }


def measure_repeated(
    fn: Callable[[], None],
    *,
    warmup: int = 1,
    runs: int = 5,
) -> TimingStats:
    """Time ``fn`` with warm-up iterations excluded from statistics."""
    for _ in range(max(warmup, 0)):
        fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return TimingStats(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        min=float(arr.min()),
        max=float(arr.max()),
        runs=times,
    )


def slowdown(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator, or None if denominator is non-positive."""
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON results under ``path``, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def nanonet_mlp(sizes: Sequence[int]):
    """Build a NanoNet MLP: sizes[0] -> ... -> sizes[-1] with ReLU between Denses."""
    from nanonet import Sequential
    from nanonet.layers import Dense, ReLU

    layers: list[Any] = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    return Sequential(layers)


def pytorch_mlp(sizes: Sequence[int], *, dtype=None):
    """Build an equivalent PyTorch MLP (Linear + ReLU)."""
    require_torch()
    import torch.nn as nn

    modules: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        modules.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            modules.append(nn.ReLU())
    model = nn.Sequential(*modules)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def copy_nanonet_weights_to_pytorch(nn_model, pt_model) -> None:
    """Copy Dense weights from NanoNet into matching PyTorch Linear layers.

    NanoNet stores weights as ``(in_features, out_features)``.
    PyTorch ``Linear.weight`` is ``(out_features, in_features)``, so each
    weight matrix is transposed on copy. Biases share the same layout.
    """
    torch = require_torch()
    nn_denses = [m for m in nn_model.modules() if type(m).__name__ == "Dense"]
    pt_linears = [m for m in pt_model.modules() if type(m).__name__ == "Linear"]
    if len(nn_denses) != len(pt_linears):
        raise ValueError(
            f"Layer count mismatch: NanoNet Dense={len(nn_denses)} "
            f"vs PyTorch Linear={len(pt_linears)}"
        )
    with torch.no_grad():
        for dense, linear in zip(nn_denses, pt_linears):
            w = np.asarray(dense.weight.data, dtype=np.float64).T.copy()
            linear.weight.copy_(torch.from_numpy(w))
            if dense.bias is not None:
                if linear.bias is None:
                    raise ValueError("PyTorch Linear missing bias")
                linear.bias.copy_(torch.from_numpy(np.asarray(dense.bias.data, dtype=np.float64)))


def assign_numpy_params_to_nanonet(nn_model, params: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
    """Assign ``(weight, bias)`` pairs into NanoNet Dense layers in order.

    Weights must already be in NanoNet layout ``(in_features, out_features)``.
    """
    denses = [m for m in nn_model.modules() if type(m).__name__ == "Dense"]
    if len(denses) != len(params):
        raise ValueError(f"Expected {len(denses)} param pairs, got {len(params)}")
    for dense, (w, b) in zip(denses, params):
        dense.weight.data = np.asarray(w, dtype=np.float64).copy()
        if b is not None:
            if dense.bias is None:
                raise ValueError("Dense layer has no bias")
            dense.bias.data = np.asarray(b, dtype=np.float64).copy()


def extract_nanonet_params(nn_model) -> list[tuple[str, np.ndarray]]:
    """Return ``(name, array)`` for each Dense weight/bias in order."""
    out: list[tuple[str, np.ndarray]] = []
    for i, m in enumerate(m for m in nn_model.modules() if type(m).__name__ == "Dense"):
        out.append((f"layer{i + 1}.weight", np.asarray(m.weight.data, dtype=np.float64).copy()))
        if m.bias is not None:
            out.append((f"layer{i + 1}.bias", np.asarray(m.bias.data, dtype=np.float64).copy()))
    return out


def extract_pytorch_params(pt_model) -> list[tuple[str, np.ndarray]]:
    """Return NanoNet-oriented ``(name, array)`` copies of PyTorch Linear params."""
    out: list[tuple[str, np.ndarray]] = []
    linears = [m for m in pt_model.modules() if type(m).__name__ == "Linear"]
    for i, linear in enumerate(linears):
        # Transpose back to NanoNet (in, out) layout for direct comparison.
        w = linear.weight.detach().cpu().numpy().astype(np.float64).T
        out.append((f"layer{i + 1}.weight", w.copy()))
        if linear.bias is not None:
            b = linear.bias.detach().cpu().numpy().astype(np.float64)
            out.append((f"layer{i + 1}.bias", b.copy()))
    return out


def extract_nanonet_grads(nn_model) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for i, m in enumerate(m for m in nn_model.modules() if type(m).__name__ == "Dense"):
        if m.weight.grad is None:
            raise RuntimeError(f"Missing grad for Dense layer {i + 1} weight")
        out.append((f"layer{i + 1}.weight", np.asarray(m.weight.grad, dtype=np.float64).copy()))
        if m.bias is not None:
            if m.bias.grad is None:
                raise RuntimeError(f"Missing grad for Dense layer {i + 1} bias")
            out.append((f"layer{i + 1}.bias", np.asarray(m.bias.grad, dtype=np.float64).copy()))
    return out


def extract_pytorch_grads(pt_model) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    linears = [m for m in pt_model.modules() if type(m).__name__ == "Linear"]
    for i, linear in enumerate(linears):
        if linear.weight.grad is None:
            raise RuntimeError(f"Missing grad for Linear layer {i + 1} weight")
        w_grad = linear.weight.grad.detach().cpu().numpy().astype(np.float64).T
        out.append((f"layer{i + 1}.weight", w_grad.copy()))
        if linear.bias is not None:
            if linear.bias.grad is None:
                raise RuntimeError(f"Missing grad for Linear layer {i + 1} bias")
            b_grad = linear.bias.grad.detach().cpu().numpy().astype(np.float64)
            out.append((f"layer{i + 1}.bias", b_grad.copy()))
    return out
