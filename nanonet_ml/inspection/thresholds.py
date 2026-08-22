"""Default thresholds for NanoNet model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Conservative defaults for heuristic diagnostics.

    Finite-value checks (NaN / Inf) are definitive and do not use these
    thresholds. Magnitude and ratio checks are intentionally conservative to
    limit false positives during normal initialization and training.
    """

    # Gradient L2 norms
    exploding_gradient_norm: float = 1e3
    vanishing_gradient_norm: float = 1e-8
    gradient_imbalance_ratio: float = 100.0

    # Activations
    dead_relu_zero_fraction: float = 0.95
    saturation_boundary: float = 0.01
    saturation_fraction: float = 0.95
    activation_std_min: float = 1e-12
    activation_min_elements: int = 8
    activation_abs_max: float = 1e6

    # Parameters
    parameter_abs_max: float = 1e6


DEFAULT_THRESHOLDS = DiagnosticThresholds()
