"""Collect structured model inspection data without permanent instrumentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nanonet.inspection.formatter import format_inspection_report
from nanonet.inspection.report import (
    ActivationStats,
    GradientStats,
    LayerInspection,
    ModelInspectionReport,
)
from nanonet.nn.module import Module
from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor, no_grad


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


def _count_params(module: Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    seen: set[int] = set()
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        n = int(param.size)
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def _parameter_memory_bytes(module: Module) -> int:
    total = 0
    seen: set[int] = set()
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.data.nbytes)
    return total


def _activation_stats(value: Any) -> ActivationStats:
    if not isinstance(value, Tensor):
        return ActivationStats(available=False)
    arr = np.asarray(value.data)
    if arr.size == 0:
        return ActivationStats(available=False)
    return ActivationStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        zero_fraction=float(np.mean(arr == 0)),
        available=True,
    )


def _shape_of(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    return None


def _gradient_stats(
    named_params: list[tuple[str, Parameter]],
) -> tuple[list[GradientStats], bool]:
    rows: list[GradientStats] = []
    any_grad = False
    for name, param in named_params:
        if param.grad is None:
            rows.append(GradientStats(name=name, exists=False))
            continue
        any_grad = True
        g = np.asarray(param.grad, dtype=np.float64)
        rows.append(
            GradientStats(
                name=name,
                exists=True,
                norm=float(np.linalg.norm(g)),
                mean=float(np.mean(g)),
                std=float(np.std(g)),
                min=float(np.min(g)),
                max=float(np.max(g)),
            )
        )
    return rows, any_grad


def _first_tensor_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    if kwargs:
        return next(iter(kwargs.values()))
    return None


def _run_instrumented_forward(
    model: Module,
    leaves: list[tuple[str, Module]],
    x: Any,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Temporarily wrap leaf ``forward`` methods, run ``model(x)``, then restore."""
    records: dict[str, dict[str, Any]] = {}
    originals: dict[int, Callable[..., Any]] = {}
    had_instance_forward: dict[int, bool] = {}

    for name, mod in leaves:
        mid = id(mod)
        had_instance_forward[mid] = "forward" in mod.__dict__
        originals[mid] = mod.forward

        def _make_wrapper(
            layer_name: str,
            original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                inp = _first_tensor_arg(args, kwargs)
                out = original(*args, **kwargs)
                records[layer_name] = {
                    "input": inp,
                    "output": out,
                    "input_shape": _shape_of(inp),
                    "output_shape": _shape_of(out),
                    "activation": _activation_stats(out),
                }
                return out

            return wrapped

        mod.forward = _make_wrapper(name, originals[mid])  # type: ignore[method-assign]

    try:
        with no_grad():
            output = model(x)
    finally:
        for _name, mod in leaves:
            mid = id(mod)
            if had_instance_forward[mid]:
                mod.forward = originals[mid]  # type: ignore[method-assign]
            elif "forward" in mod.__dict__:
                del mod.forward

    return output, records


def inspect_model(
    model: Module,
    x: Any | None = None,
    *,
    verbose: bool = True,
) -> ModelInspectionReport:
    """Inspect ``model`` structure and optionally run a safe forward pass on ``x``.

    Args:
        model: Root module to inspect.
        x: Optional sample input for runtime shapes and activation statistics.
        verbose: If True, print the formatted report to stdout.

    Returns:
        A :class:`ModelInspectionReport` independent of terminal formatting.

    Notes:
        The forward pass (when ``x`` is provided) runs under ``no_grad()`` and
        does not update parameters, clear gradients, or leave hooks installed.
        Genuine forward errors (e.g. shape mismatches) propagate normally.
    """
    leaves = leaf_modules(model)

    layers: list[LayerInspection] = []
    for name, mod in leaves:
        total, trainable = _count_params(mod)
        param_names = [n for n, _ in mod.named_parameters()]
        layers.append(
            LayerInspection(
                name=name,
                type=type(mod).__name__,
                parameter_count=total,
                trainable_parameter_count=trainable,
                parameter_names=param_names,
            )
        )

    total_params, trainable_params = _count_params(model)
    report = ModelInspectionReport(
        model_name=type(model).__name__,
        model_type=type(model).__name__,
        layers=layers,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        non_trainable_parameters=total_params - trainable_params,
        estimated_parameter_memory_bytes=_parameter_memory_bytes(model),
    )

    grads, grads_available = _gradient_stats(model.named_parameters())
    report.gradients = grads
    report.gradients_available = grads_available

    if x is not None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        report.input_shape = tuple(x.shape)

        was_training = model.training
        try:
            output, records = _run_instrumented_forward(model, leaves, x)
        finally:
            model.train(was_training)

        report.output_shape = _shape_of(output)
        report.runtime_captured = True

        for layer in report.layers:
            rec = records.get(layer.name)
            if rec is None:
                continue
            layer.input_shape = rec["input_shape"]
            layer.output_shape = rec["output_shape"]
            layer.activation = rec["activation"]
            out_val = rec["output"]
            if isinstance(out_val, Tensor):
                layer.output_tensor_id = id(out_val)
            inp_val = rec["input"]
            if isinstance(inp_val, Tensor):
                layer.input_tensor_id = id(inp_val)

    if verbose:
        print(format_inspection_report(report), end="")

    return report
