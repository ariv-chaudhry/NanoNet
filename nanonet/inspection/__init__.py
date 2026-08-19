"""Model inspection and explainability utilities."""

from nanonet.inspection.formatter import format_bytes, format_inspection_report
from nanonet.inspection.inspector import inspect_model, leaf_modules
from nanonet.inspection.report import (
    ActivationStats,
    GradientStats,
    LayerInspection,
    ModelInspectionReport,
)

__all__ = [
    "ActivationStats",
    "GradientStats",
    "LayerInspection",
    "ModelInspectionReport",
    "format_bytes",
    "format_inspection_report",
    "inspect_model",
    "leaf_modules",
]
