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
from nanonet.inspection.utils import (
    activation_stats,
    count_parameters,
    first_arg,
    leaf_modules,
    parameter_memory_bytes,
    shape_of,
)
from nanonet.nn.module import Module
from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor, no_grad

# Re-export for backward-compatible imports from inspector.
__all__ = [
    "inspect_model",
    "leaf_modules",
    "iter_named_modules",
]


def iter_named_modules(
    module: Module,
    prefix: str = "",
    *,
    include_root: bool = True,
) -> list[tuple[str, Module]]:
    from nanonet.inspection.utils import iter_named_modules as _iter

    return _iter(module, prefix, include_root=include_root)


def _activation_stats(value: Any) -> ActivationStats:
    return activation_stats(value)


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
                inp = first_arg(args, kwargs)
                out = original(*args, **kwargs)
                records[layer_name] = {
                    "input": inp,
                    "output": out,
                    "input_shape": shape_of(inp),
                    "output_shape": shape_of(out),
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
        total, trainable = count_parameters(mod)
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

    total_params, trainable_params = count_parameters(model)
    report = ModelInspectionReport(
        model_name=type(model).__name__,
        model_type=type(model).__name__,
        layers=layers,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        non_trainable_parameters=total_params - trainable_params,
        estimated_parameter_memory_bytes=parameter_memory_bytes(model),
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

        report.output_shape = shape_of(output)
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
