"""Collect structured model inspection data without permanent instrumentation."""

from __future__ import annotations

from typing import Any

from nanonet.inspection.formatter import format_inspection_report
from nanonet.inspection.instrumentation import run_observed_forward
from nanonet.inspection.report import LayerInspection, ModelInspectionReport
from nanonet.inspection.utils import (
    activation_stats,
    count_parameters,
    gradient_stats_rows,
    leaf_modules,
    parameter_memory_bytes,
    shape_of,
)
from nanonet.nn.module import Module
from nanonet.tensor import Tensor

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
        Shared leaf modules are instrumented once. Genuine forward errors
        (e.g. shape mismatches) propagate normally after cleanup.
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

    grads, grads_available = gradient_stats_rows(model.named_parameters())
    report.gradients = grads
    report.gradients_available = grads_available

    if x is not None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        report.input_shape = tuple(x.shape)

        records: dict[str, dict[str, Any]] = {}

        def on_call(
            layer_name: str,
            _module: Module,
            inp: Any,
            out: Any,
            _call_index: int,
        ) -> None:
            # Last call wins for structural inspect (shared modules overwrite).
            records[layer_name] = {
                "input": inp,
                "output": out,
                "input_shape": shape_of(inp),
                "output_shape": shape_of(out),
                "activation": activation_stats(out),
            }

        output = run_observed_forward(model, x, on_call, disable_grad=True)
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
