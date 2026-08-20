"""Plain-text formatting for inspection reports and execution traces."""

from __future__ import annotations

from nanonet.inspection.report import ModelInspectionReport, ModelTrace, TensorTraceInfo
from nanonet.inspection.utils import format_duration


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as a compact human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"


def format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return str(shape)


def format_inspection_report(report: ModelInspectionReport) -> str:
    """Render an inspection report as terminal-safe plain text."""
    lines: list[str] = []
    width = 72
    rule = "-" * width

    lines.append("NanoNet Model Inspector")
    lines.append(rule)
    lines.append("")
    lines.append(f"Model: {report.model_type}")
    lines.append(f"Layers: {report.layer_count}")
    lines.append(f"Parameters: {_fmt_int(report.total_parameters)}")
    lines.append(f"Trainable: {_fmt_int(report.trainable_parameters)}")
    if report.non_trainable_parameters:
        lines.append(
            f"Non-trainable: {_fmt_int(report.non_trainable_parameters)}"
        )
    lines.append(
        f"Parameter Memory: {format_bytes(report.estimated_parameter_memory_bytes)}"
    )
    if report.runtime_captured:
        lines.append(f"Input Shape: {format_shape(report.input_shape)}")
        lines.append(f"Output Shape: {format_shape(report.output_shape)}")

    lines.append("")
    lines.append("Layers")
    lines.append(rule)
    lines.append(f"{'Name':<24} {'Type':<16} {'Parameters':>12}")
    lines.append(rule)
    for layer in report.layers:
        lines.append(
            f"{layer.name:<24} {layer.type:<16} "
            f"{_fmt_int(layer.parameter_count):>12}"
        )
    lines.append(rule)
    lines.append(
        f"{'Total':<24} {'':<16} {_fmt_int(report.total_parameters):>12}"
    )

    if report.runtime_captured:
        lines.append("")
        lines.append("Forward Pass")
        lines.append(rule)
        lines.append(
            f"{'Layer':<24} {'Input Shape':<18} {'Output Shape':<18}"
        )
        lines.append(rule)
        for layer in report.layers:
            lines.append(
                f"{layer.name:<24} "
                f"{format_shape(layer.input_shape):<18} "
                f"{format_shape(layer.output_shape):<18}"
            )

        lines.append("")
        lines.append("Activation Statistics")
        lines.append(rule)
        lines.append(
            f"{'Layer':<16} {'Mean':>10} {'Std':>10} "
            f"{'Min':>10} {'Max':>10} {'Zero %':>10}"
        )
        lines.append(rule)
        for layer in report.layers:
            stats = layer.activation
            if not stats.available:
                lines.append(
                    f"{layer.name:<16} {'n/a':>10} {'n/a':>10} "
                    f"{'n/a':>10} {'n/a':>10} {'n/a':>10}"
                )
                continue
            zero_pct = (
                100.0 * stats.zero_fraction
                if stats.zero_fraction is not None
                else float("nan")
            )
            lines.append(
                f"{layer.name:<16} "
                f"{_fmt_float(stats.mean):>10} "
                f"{_fmt_float(stats.std):>10} "
                f"{_fmt_float(stats.min):>10} "
                f"{_fmt_float(stats.max):>10} "
                f"{zero_pct:>9.1f}%"
            )

    lines.append("")
    lines.append("Gradient Summary")
    lines.append(rule)
    if not report.gradients_available:
        lines.append("Gradients: not available (backward has not been run)")
    else:
        lines.append(f"{'Parameter':<28} {'Norm':>12}")
        lines.append(rule)
        for g in report.gradients:
            if not g.exists or g.norm is None:
                lines.append(f"{g.name:<28} {'-':>12}")
            else:
                lines.append(f"{g.name:<28} {g.norm:>12.6f}")

    lines.append("")
    return "\n".join(lines)


def format_execution_trace(trace: ModelTrace) -> str:
    """Render a chronological execution trace as terminal-safe plain text."""
    lines: list[str] = []
    width = 72
    rule = "-" * width

    lines.append("NanoNet Execution Trace")
    lines.append(rule)
    lines.append("")
    lines.append(f"Model: {trace.model_type}")
    lines.append("")
    lines.append("Input")
    if not trace.inputs:
        lines.append("  (no Tensor inputs recorded)")
    else:
        for info in trace.inputs:
            lines.append(f"  {_format_tensor_line(info)}")

    for step in trace.steps:
        lines.append("")
        title = f"Step {step.index} - {step.module_name} | {step.module_type}"
        if step.call_index > 1:
            title += f" (call {step.call_index})"
        lines.append(title)
        lines.append(rule)
        lines.append("Input")
        if not step.inputs:
            lines.append("  (none / non-Tensor)")
        else:
            for info in step.inputs:
                lines.append(f"  {_format_tensor_line(info, compact=True)}")
        lines.append("Output")
        if not step.outputs:
            lines.append("  (none / non-Tensor)")
        else:
            for info in step.outputs:
                lines.append(f"  {_format_tensor_line(info, compact=True)}")
        lines.append(f"Parameters: {_fmt_int(step.parameter_count)}")
        lines.append(f"Time: {format_duration(step.duration_seconds)}")

    lines.append("")
    lines.append("Summary")
    lines.append(rule)
    lines.append(f"Steps: {len(trace.steps)}")
    if trace.outputs:
        out_desc = ", ".join(
            _format_tensor_line(info, compact=True) for info in trace.outputs
        )
        lines.append(f"Output: {out_desc}")
    else:
        lines.append("Output: (non-Tensor / unavailable)")
    lines.append(
        f"Forward duration: {format_duration(trace.forward_duration_seconds)}"
    )
    lines.append(
        f"Traced module time: {format_duration(trace.traced_duration_seconds)}"
    )
    lines.append("")
    lines.append(
        "Note: timings include tracing instrumentation overhead and are for "
        "debugging, not benchmarking."
    )
    lines.append("")
    return "\n".join(lines)


def _format_tensor_line(info: TensorTraceInfo, *, compact: bool = False) -> str:
    shape = format_shape(info.shape)
    if compact:
        return f"{info.trace_id} {shape}"
    dtype = info.dtype or "?"
    rg = info.requires_grad
    rg_s = f" requires_grad={rg}" if rg is not None else ""
    return f"{info.trace_id}   shape={shape} dtype={dtype}{rg_s}"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
