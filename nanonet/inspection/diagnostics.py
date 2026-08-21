"""Model diagnostics engine: evidence-based numerical / training health checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from nanonet.inspection.checks import build_context, run_all_checks
from nanonet.inspection.formatter import format_diagnostics_report
from nanonet.inspection.instrumentation import run_observed_forward
from nanonet.inspection.report import (
    ActivationStats,
    DiagnosticsReport,
    RuntimeActivationRecord,
)
from nanonet.inspection.thresholds import DEFAULT_THRESHOLDS, DiagnosticThresholds
from nanonet.inspection.utils import activation_stats
from nanonet.nn.module import Module
from nanonet.tensor import Tensor


def _saturation_fraction(
    arr: np.ndarray,
    module_type: str,
    boundary: float,
) -> float | None:
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if module_type == "Sigmoid":
        sat = (finite < boundary) | (finite > 1.0 - boundary)
    elif module_type == "Tanh":
        sat = (finite < -1.0 + boundary) | (finite > 1.0 - boundary)
    else:
        return None
    return float(np.mean(sat))


def _enrich_activation_stats(
    stats: ActivationStats,
    value: Any,
    module_type: str,
    thresholds: DiagnosticThresholds,
) -> ActivationStats:
    if not isinstance(value, Tensor) or not stats.available:
        return stats
    sat = _saturation_fraction(
        np.asarray(value.data),
        module_type,
        thresholds.saturation_boundary,
    )
    stats.saturation_fraction = sat
    return stats


def _collect_activations(
    model: Module,
    x: Any,
    thresholds: DiagnosticThresholds,
) -> list[RuntimeActivationRecord]:
    """One instrumented ``no_grad`` forward; restores wrappers even on failure."""
    records: list[RuntimeActivationRecord] = []

    def on_call(
        layer_name: str,
        module_ref: Module,
        _inp: Any,
        out: Any,
        call_index: int,
    ) -> None:
        mtype = type(module_ref).__name__
        stats = activation_stats(out)
        stats = _enrich_activation_stats(stats, out, mtype, thresholds)
        records.append(
            RuntimeActivationRecord(
                name=layer_name,
                module_type=mtype,
                call_index=call_index,
                stats=stats,
            )
        )

    run_observed_forward(model, x, on_call, disable_grad=True)
    return records


def diagnose_model(
    model: Module,
    x: Any | None = None,
    *,
    verbose: bool = True,
    thresholds: DiagnosticThresholds | None = None,
) -> DiagnosticsReport:
    """Diagnose potential numerical / training issues in ``model``.

    Analyzes parameters and currently available gradients. When ``x`` is given,
    also runs a single ``no_grad`` forward pass to inspect activations.

    Does not call ``backward()``, clear gradients, or modify parameters.
    Finite checks (NaN/Inf) are definitive; magnitude checks are heuristic.

    Args:
        model: Root module to diagnose.
        x: Optional sample input for activation diagnostics.
        verbose: If True, print the formatted report to stdout.
        thresholds: Optional :class:`DiagnosticThresholds` overrides.

    Returns:
        A :class:`DiagnosticsReport`.
    """
    thr = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    activations: list[RuntimeActivationRecord] = []
    activations_analyzed = False

    if x is not None:
        activations = _collect_activations(model, x, thr)
        activations_analyzed = True

    ctx = build_context(
        model,
        activations,
        thr,
        activations_analyzed=activations_analyzed,
    )
    findings = run_all_checks(ctx)

    warnings = sum(1 for f in findings if f.severity == "warning")
    critical = sum(1 for f in findings if f.severity == "critical")
    info = sum(1 for f in findings if f.severity == "info")

    report = DiagnosticsReport(
        model_name=type(model).__name__,
        findings=findings,
        checks_run=ctx.checks_run,
        warnings=warnings,
        critical=critical,
        info=info,
        activations_analyzed=activations_analyzed,
        gradients_available=ctx.gradients_available,
    )

    if verbose:
        print(format_diagnostics_report(report), end="")

    return report
