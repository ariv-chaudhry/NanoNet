"""Individual diagnostic checks over a shared context."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nanonet_ml.inspection.report import DiagnosticFinding, RuntimeActivationRecord
from nanonet_ml.inspection.thresholds import DiagnosticThresholds
from nanonet_ml.inspection.utils import gradient_norm, leaf_modules
from nanonet_ml.nn.module import Module
from nanonet_ml.nn.parameter import Parameter


@dataclass
class ParamInfo:
    name: str
    param: Parameter
    abs_max: float | None
    has_nan: bool
    has_inf: bool
    grad_norm: float | None
    grad_has_nan: bool
    grad_has_inf: bool
    grad_exists: bool


@dataclass
class LayerGradInfo:
    name: str
    norm: float


@dataclass
class DiagnosticContext:
    """Precomputed model state shared by all diagnostic checks."""

    model: Module
    thresholds: DiagnosticThresholds
    parameters: list[ParamInfo]
    layer_gradients: list[LayerGradInfo]
    activations: list[RuntimeActivationRecord]
    activations_analyzed: bool
    gradients_available: bool
    leaf_types: dict[str, str] = field(default_factory=dict)
    checks_run: int = 0


def build_context(
    model: Module,
    activations: list[RuntimeActivationRecord],
    thresholds: DiagnosticThresholds,
    *,
    activations_analyzed: bool,
) -> DiagnosticContext:
    """Collect parameter / gradient metadata once for all checks."""
    params: list[ParamInfo] = []
    seen: set[int] = set()
    any_grad = False

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        data = np.asarray(param.data)
        has_nan = bool(np.any(np.isnan(data))) if data.size else False
        has_inf = bool(np.any(np.isinf(data))) if data.size else False
        finite = data[np.isfinite(data)]
        abs_max = float(np.max(np.abs(finite))) if finite.size else None

        grad_exists = param.grad is not None
        grad_has_nan = False
        grad_has_inf = False
        gnorm: float | None = None
        if grad_exists:
            any_grad = True
            g = np.asarray(param.grad)
            grad_has_nan = bool(np.any(np.isnan(g))) if g.size else False
            grad_has_inf = bool(np.any(np.isinf(g))) if g.size else False
            if g.size:
                if np.all(np.isfinite(g)):
                    gnorm = gradient_norm(g)
                elif np.any(np.isfinite(g)):
                    gnorm = gradient_norm(g[np.isfinite(g)])

        params.append(
            ParamInfo(
                name=name,
                param=param,
                abs_max=abs_max,
                has_nan=has_nan,
                has_inf=has_inf,
                grad_norm=gnorm,
                grad_has_nan=grad_has_nan,
                grad_has_inf=grad_has_inf,
                grad_exists=grad_exists,
            )
        )

    leaves = leaf_modules(model)
    leaf_types = {name: type(mod).__name__ for name, mod in leaves}
    layer_grads: list[LayerGradInfo] = []
    for leaf_name, mod in leaves:
        sq = 0.0
        counted = False
        seen_local: set[int] = set()
        for _pname, p in mod.named_parameters():
            if id(p) in seen_local:
                continue
            seen_local.add(id(p))
            if p.grad is None:
                continue
            g = np.asarray(p.grad)
            if g.size == 0:
                continue
            if np.all(np.isfinite(g)):
                n = gradient_norm(g)
            elif np.any(np.isfinite(g)):
                n = gradient_norm(g[np.isfinite(g)])
            else:
                continue
            if n is None:
                continue
            sq += n * n
            counted = True
        if counted:
            layer_grads.append(LayerGradInfo(name=leaf_name, norm=float(np.sqrt(sq))))

    return DiagnosticContext(
        model=model,
        thresholds=thresholds,
        parameters=params,
        layer_gradients=layer_grads,
        activations=activations,
        activations_analyzed=activations_analyzed,
        gradients_available=any_grad,
        leaf_types=leaf_types,
    )


def _finding(
    *,
    severity: str,
    category: str,
    code: str,
    message: str,
    target: str | None = None,
    observed_value: float | None = None,
    threshold: float | None = None,
    explanation: str | None = None,
    recommendation: str | None = None,
) -> DiagnosticFinding:
    return DiagnosticFinding(
        severity=severity,
        category=category,
        code=code,
        message=message,
        target=target,
        observed_value=observed_value,
        threshold=threshold,
        explanation=explanation,
        recommendation=recommendation,
    )


def check_parameters(ctx: DiagnosticContext) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    thr = ctx.thresholds
    ctx.checks_run += 3

    for info in ctx.parameters:
        if info.has_nan:
            findings.append(
                _finding(
                    severity="critical",
                    category="parameters",
                    code="PARAM_NAN",
                    message=f"NaN values detected in parameter '{info.name}'.",
                    target=info.name,
                    explanation=(
                        "Non-finite parameters usually indicate prior overflow "
                        "or an invalid update."
                    ),
                    recommendation=(
                        "Inspect the first operation producing non-finite values "
                        "and verify learning rate / input scaling."
                    ),
                )
            )
        if info.has_inf:
            findings.append(
                _finding(
                    severity="critical",
                    category="parameters",
                    code="PARAM_INF",
                    message=f"Infinite values detected in parameter '{info.name}'.",
                    target=info.name,
                    explanation="Infinite parameters indicate numerical blow-up.",
                    recommendation=(
                        "Inspect the first operation producing non-finite values "
                        "and verify learning rate / input scaling."
                    ),
                )
            )
        if (
            info.abs_max is not None
            and info.abs_max > thr.parameter_abs_max
            and not info.has_nan
            and not info.has_inf
        ):
            findings.append(
                _finding(
                    severity="warning",
                    category="parameters",
                    code="PARAM_LARGE",
                    message=f"Parameter '{info.name}' has unusually large magnitude.",
                    target=info.name,
                    observed_value=info.abs_max,
                    threshold=thr.parameter_abs_max,
                    explanation=(
                        f"Observed abs max {info.abs_max:.4g} exceeds threshold "
                        f"{thr.parameter_abs_max:g}."
                    ),
                    recommendation=(
                        "Consider reducing the learning rate or checking for unstable updates."
                    ),
                )
            )
    return findings


def check_gradients(ctx: DiagnosticContext) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    thr = ctx.thresholds
    ctx.checks_run += 5

    if not ctx.gradients_available:
        findings.append(
            _finding(
                severity="info",
                category="gradients",
                code="GRAD_UNAVAILABLE",
                message=(
                    "Gradient diagnostics skipped: no parameter gradients are present."
                ),
                explanation=(
                    "Call loss.backward() before diagnose() to analyze gradients. "
                    "diagnose() never runs backward automatically."
                ),
            )
        )
        return findings

    trainable = [p for p in ctx.parameters if p.param.requires_grad]
    for info in trainable:
        if info.grad_has_nan:
            findings.append(
                _finding(
                    severity="critical",
                    category="gradients",
                    code="GRAD_NAN",
                    message=f"NaN gradient detected for '{info.name}'.",
                    target=info.name,
                    recommendation=(
                        "Inspect the first operation producing non-finite values "
                        "and verify learning rate / input scaling."
                    ),
                )
            )
        if info.grad_has_inf:
            findings.append(
                _finding(
                    severity="critical",
                    category="gradients",
                    code="GRAD_INF",
                    message=f"Infinite gradient detected for '{info.name}'.",
                    target=info.name,
                    recommendation=(
                        "Consider reducing the learning rate or applying gradient clipping."
                    ),
                )
            )
        if info.grad_exists and info.grad_norm is not None:
            if info.grad_norm > thr.exploding_gradient_norm:
                findings.append(
                    _finding(
                        severity="warning",
                        category="gradients",
                        code="GRAD_EXPLODING",
                        message=f"Gradient norm for '{info.name}' is very large.",
                        target=info.name,
                        observed_value=info.grad_norm,
                        threshold=thr.exploding_gradient_norm,
                        explanation=(
                            f"Observed L2 norm {info.grad_norm:.4g} exceeds "
                            f"{thr.exploding_gradient_norm:g}."
                        ),
                        recommendation=(
                            "Consider reducing the learning rate or applying "
                            "gradient clipping."
                        ),
                    )
                )
            elif 0.0 < info.grad_norm < thr.vanishing_gradient_norm:
                findings.append(
                    _finding(
                        severity="warning",
                        category="gradients",
                        code="GRAD_VANISHING",
                        message=f"Gradient norm for '{info.name}' is extremely small.",
                        target=info.name,
                        observed_value=info.grad_norm,
                        threshold=thr.vanishing_gradient_norm,
                        explanation=(
                            f"Observed L2 norm {info.grad_norm:.4g} is below "
                            f"{thr.vanishing_gradient_norm:g}."
                        ),
                        recommendation=(
                            "Consider checking activation saturation, "
                            "initialization, or network depth."
                        ),
                    )
                )

    for info in trainable:
        if not info.grad_exists:
            findings.append(
                _finding(
                    severity="warning",
                    category="gradients",
                    code="GRAD_MISSING",
                    message=(
                        f"Parameter '{info.name}' has no gradient while other "
                        "trainable parameters do."
                    ),
                    target=info.name,
                    explanation=(
                        "This may indicate a disconnected parameter that does not "
                        "participate in the computed loss."
                    ),
                    recommendation=(
                        "Confirm the parameter participates in the forward path "
                        "used for the loss."
                    ),
                )
            )

    for leaf_name, mod in leaf_modules(ctx.model):
        named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
        if not named:
            continue
        norms: list[float | None] = []
        for _n, p in named:
            if p.grad is None:
                norms.append(None)
            else:
                norms.append(gradient_norm(p.grad))
        if norms and all(g is not None for g in norms) and all(g == 0.0 for g in norms):
            findings.append(
                _finding(
                    severity="warning",
                    category="gradients",
                    code="GRAD_ZERO_LAYER",
                    message=(
                        f"All parameter gradients in layer '{leaf_name}' are exactly zero."
                    ),
                    target=leaf_name,
                    observed_value=0.0,
                    explanation=(
                        "Exact zeros can be expected for sparse losses or inactive "
                        "units, but an entire layer at zero may indicate a dead path."
                    ),
                    recommendation=(
                        "Inspect activations and whether this layer contributes to the loss."
                    ),
                )
            )

    finite_layers = [lg for lg in ctx.layer_gradients if lg.norm > 0]
    if len(finite_layers) >= 2:
        max_lg = max(finite_layers, key=lambda x: x.norm)
        min_lg = min(finite_layers, key=lambda x: x.norm)
        ratio = max_lg.norm / min_lg.norm
        if ratio >= thr.gradient_imbalance_ratio:
            findings.append(
                _finding(
                    severity="warning",
                    category="gradients",
                    code="GRAD_IMBALANCE",
                    message=(
                        f"Gradient norms differ substantially between layers "
                        f"'{min_lg.name}' and '{max_lg.name}'."
                    ),
                    target=f"{min_lg.name} vs {max_lg.name}",
                    observed_value=ratio,
                    threshold=thr.gradient_imbalance_ratio,
                    explanation=(
                        f"Layer '{min_lg.name}' norm={min_lg.norm:.4g}; "
                        f"'{max_lg.name}' norm={max_lg.norm:.4g}; "
                        f"ratio={ratio:.4g}x "
                        f"(threshold {thr.gradient_imbalance_ratio:g}x). "
                        "This may indicate vanishing or exploding gradients across depth."
                    ),
                    recommendation=(
                        "Consider checking activation saturation, "
                        "initialization, or network depth."
                    ),
                )
            )

    return findings


def check_activations(ctx: DiagnosticContext) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    thr = ctx.thresholds

    if not ctx.activations_analyzed:
        ctx.checks_run += 1
        findings.append(
            _finding(
                severity="info",
                category="activations",
                code="ACT_SKIPPED",
                message="Activation diagnostics skipped: no sample input provided.",
                explanation="Call model.diagnose(x) to analyze runtime activations.",
            )
        )
        return findings

    ctx.checks_run += 5
    first_nan: str | None = None
    first_inf: str | None = None

    for rec in ctx.activations:
        stats = rec.stats
        if not stats.available:
            continue
        target = rec.display_name

        if stats.nan_count > 0:
            if first_nan is None:
                first_nan = target
                findings.append(
                    _finding(
                        severity="critical",
                        category="activations",
                        code="ACT_NAN",
                        message=f"NaN values first appear in output of '{target}'.",
                        target=target,
                        observed_value=float(stats.nan_count),
                        explanation=(
                            "Later layers may also contain NaNs as a consequence "
                            "of this layer."
                        ),
                        recommendation=(
                            "Inspect the first operation producing non-finite values "
                            "and verify learning rate / input scaling."
                        ),
                    )
                )
            continue

        if stats.inf_count > 0:
            if first_inf is None:
                first_inf = target
                findings.append(
                    _finding(
                        severity="critical",
                        category="activations",
                        code="ACT_INF",
                        message=f"Infinite values first appear in output of '{target}'.",
                        target=target,
                        observed_value=float(stats.inf_count),
                        explanation=(
                            "Later layers may also contain infinities as a "
                            "consequence of this layer."
                        ),
                        recommendation=(
                            "Consider reducing the learning rate or checking for "
                            "unstable operations."
                        ),
                    )
                )
            continue

        if stats.abs_max is not None and stats.abs_max > thr.activation_abs_max:
            findings.append(
                _finding(
                    severity="warning",
                    category="activations",
                    code="ACT_LARGE",
                    message=f"Activation magnitude in '{target}' is unusually large.",
                    target=target,
                    observed_value=stats.abs_max,
                    threshold=thr.activation_abs_max,
                    recommendation=(
                        "Consider checking input scaling or layer initialization."
                    ),
                )
            )

        if (
            stats.std is not None
            and stats.element_count >= thr.activation_min_elements
            and stats.std < thr.activation_std_min
        ):
            # Dead ReLU already covers the all-zero case more specifically.
            if not (
                rec.module_type == "ReLU"
                and stats.zero_fraction is not None
                and stats.zero_fraction >= thr.dead_relu_zero_fraction
            ):
                findings.append(
                    _finding(
                        severity="warning",
                        category="activations",
                        code="ACT_CONSTANT",
                        message=(
                            f"Activation output of '{target}' has near-zero variance."
                        ),
                        target=target,
                        observed_value=stats.std,
                        threshold=thr.activation_std_min,
                        explanation=(
                            f"std={stats.std:.4g} over {stats.element_count} elements "
                            f"(min elements for this check: {thr.activation_min_elements})."
                        ),
                        recommendation=(
                            "Inspect whether this layer has collapsed to a "
                            "near-constant output."
                        ),
                    )
                )

        if rec.module_type == "ReLU" and stats.zero_fraction is not None:
            if stats.zero_fraction >= thr.dead_relu_zero_fraction:
                findings.append(
                    _finding(
                        severity="warning",
                        category="activations",
                        code="RELU_DEAD",
                        message=(
                            f"ReLU '{target}' has a very high fraction of zero activations."
                        ),
                        target=target,
                        observed_value=stats.zero_fraction,
                        threshold=thr.dead_relu_zero_fraction,
                        explanation=(
                            f"Observed zero fraction {stats.zero_fraction:.1%} "
                            f"(threshold {thr.dead_relu_zero_fraction:.0%}). "
                            "Approximately 50% zeros can be normal for ReLU."
                        ),
                        recommendation=(
                            "Consider inspecting initialization, learning rate, "
                            "or replacing ReLU with a non-zero-slope alternative."
                        ),
                    )
                )

        if rec.module_type in {"Sigmoid", "Tanh"}:
            sat = stats.saturation_fraction
            if sat is not None and sat >= thr.saturation_fraction:
                findings.append(
                    _finding(
                        severity="warning",
                        category="activations",
                        code="ACT_SATURATION",
                        message=(
                            f"{rec.module_type} '{target}' appears heavily saturated."
                        ),
                        target=target,
                        observed_value=sat,
                        threshold=thr.saturation_fraction,
                        explanation=(
                            f"Observed saturation fraction {sat:.1%} "
                            f"(threshold {thr.saturation_fraction:.0%})."
                        ),
                        recommendation=(
                            "Consider checking initialization or replacing saturated "
                            "activations deeper in the network."
                        ),
                    )
                )

    return findings


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _sort_findings(findings: list[DiagnosticFinding]) -> list[DiagnosticFinding]:
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_ORDER.get(f.severity, 99),
            f.category,
            f.target or "",
            f.code,
        ),
    )


def run_all_checks(ctx: DiagnosticContext) -> list[DiagnosticFinding]:
    """Run the full check suite and return sorted findings."""
    findings: list[DiagnosticFinding] = []
    findings.extend(check_parameters(ctx))
    findings.extend(check_gradients(ctx))
    findings.extend(check_activations(ctx))
    return _sort_findings(findings)
