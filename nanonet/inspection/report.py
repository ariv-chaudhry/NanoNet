"""Structured data types for NanoNet model inspection reports.

These dataclasses separate collection from presentation so later stages
(``trace``, ``diagnose``) can reuse the same records without parsing text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActivationStats:
    """Basic statistics over a tensor-valued layer output."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    zero_fraction: float | None = None  # in [0, 1]
    available: bool = False


@dataclass
class GradientStats:
    """Summary of a parameter's gradient, if present."""

    name: str
    exists: bool
    norm: float | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


@dataclass
class LayerInspection:
    """Inspection record for a single (typically leaf) module.

    Fields such as ``input_tensor_id`` / ``output_tensor_id`` are reserved for
    future ``trace()`` support and may be ``None`` in Stage 1.
    """

    name: str
    type: str
    parameter_count: int
    trainable_parameter_count: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    activation: ActivationStats = field(default_factory=ActivationStats)
    # Future-facing identifiers (unused in Stage 1 formatting).
    input_tensor_id: int | None = None
    output_tensor_id: int | None = None
    parameter_names: list[str] = field(default_factory=list)


@dataclass
class ModelInspectionReport:
    """Complete structured result of ``model.inspect()``."""

    model_name: str
    model_type: str
    layers: list[LayerInspection]
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    estimated_parameter_memory_bytes: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    gradients: list[GradientStats] = field(default_factory=list)
    gradients_available: bool = False
    runtime_captured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return len(self.layers)
