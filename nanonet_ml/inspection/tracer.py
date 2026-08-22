"""Chronological execution tracing for NanoNet models."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from nanonet_ml.inspection.formatter import format_execution_trace
from nanonet_ml.inspection.instrumentation import unique_leaf_modules
from nanonet_ml.inspection.report import ModelTrace, TensorTraceInfo, TraceStep
from nanonet_ml.inspection.utils import (
    collect_tensors,
    count_parameters,
    extract_tensor_metadata,
)
from nanonet_ml.nn.module import Module
from nanonet_ml.tensor import Tensor


class _TensorIdRegistry:
    """Map Python object identity to stable trace-local IDs (T0, T1, ...)."""

    def __init__(self) -> None:
        self._ids: dict[int, str] = {}
        self._infos: dict[str, TensorTraceInfo] = {}
        self._next = 0

    def register(self, tensor: Tensor) -> TensorTraceInfo:
        oid = id(tensor)
        if oid in self._ids:
            return self._infos[self._ids[oid]]
        tid = f"T{self._next}"
        self._next += 1
        meta = extract_tensor_metadata(tensor)
        info = TensorTraceInfo(
            trace_id=tid,
            shape=meta["shape"],
            dtype=meta["dtype"],
            requires_grad=meta["requires_grad"],
            object_id=oid,
        )
        self._ids[oid] = tid
        self._infos[tid] = info
        return info

    def register_many(self, values: list[Tensor]) -> list[TensorTraceInfo]:
        return [self.register(t) for t in values]


def _infos_from_value(value: Any, registry: _TensorIdRegistry) -> list[TensorTraceInfo]:
    tensors = collect_tensors(value)
    if not tensors:
        return []
    return registry.register_many(tensors)


def trace_model(
    model: Module,
    x: Any,
    *,
    verbose: bool = True,
) -> ModelTrace:
    """Trace a single forward pass through ``model`` with input ``x``.

    Records leaf modules in **runtime execution order**, including tensor shapes,
    trace-local tensor IDs, direct parameter counts, and per-step timing.

    Autograd remains enabled: the returned ``ModelTrace.output`` can participate
    in ``backward()``. Timing includes instrumentation overhead and is intended for
    debugging, not benchmarking.

    Args:
        model: Root module to execute.
        x: Model input (``Tensor`` or array-like).
        verbose: If True, print the formatted trace to stdout.

    Returns:
        :class:`ModelTrace` with chronological steps and the forward output.
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)

    unique_leaves = unique_leaf_modules(model)
    registry = _TensorIdRegistry()
    input_infos = _infos_from_value(x, registry)
    steps: list[TraceStep] = []
    call_counts: dict[str, int] = defaultdict(int)

    originals: dict[int, Callable[..., Any]] = {}
    had_instance_forward: dict[int, bool] = {}

    for name, mod in unique_leaves:
        mid = id(mod)
        had_instance_forward[mid] = "forward" in mod.__dict__
        originals[mid] = mod.forward

        def _make_wrapper(
            layer_name: str,
            module_ref: Module,
            original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                input_values: list[Any] = list(args) + list(kwargs.values())
                in_infos: list[TensorTraceInfo] = []
                for val in input_values:
                    in_infos.extend(_infos_from_value(val, registry))

                t0 = time.perf_counter()
                out = original(*args, **kwargs)
                dt = time.perf_counter() - t0

                out_infos = _infos_from_value(out, registry)
                param_count, _ = count_parameters(module_ref, recursive=False)
                call_counts[layer_name] += 1
                steps.append(
                    TraceStep(
                        index=len(steps) + 1,
                        module_name=layer_name,
                        module_type=type(module_ref).__name__,
                        inputs=in_infos,
                        outputs=out_infos,
                        parameter_count=param_count,
                        duration_seconds=dt,
                        call_index=call_counts[layer_name],
                        module_object_id=id(module_ref),
                    )
                )
                return out

            return wrapped

        mod.forward = _make_wrapper(name, mod, originals[mid])  # type: ignore[method-assign]

    output: Any = None
    forward_dt = 0.0
    try:
        t_forward = time.perf_counter()
        output = model(x)
        forward_dt = time.perf_counter() - t_forward
    finally:
        for _name, mod in unique_leaves:
            mid = id(mod)
            if had_instance_forward[mid]:
                mod.forward = originals[mid]  # type: ignore[method-assign]
            elif "forward" in mod.__dict__:
                del mod.forward

    output_infos = _infos_from_value(output, registry)
    traced_dt = float(sum(s.duration_seconds for s in steps))

    trace = ModelTrace(
        model_name=type(model).__name__,
        model_type=type(model).__name__,
        steps=steps,
        inputs=input_infos,
        outputs=output_infos,
        forward_duration_seconds=forward_dt,
        traced_duration_seconds=traced_dt,
        output=output,
    )

    if verbose:
        print(format_execution_trace(trace), end="")

    return trace
