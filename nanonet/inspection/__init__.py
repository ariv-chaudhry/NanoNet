"""Model inspection and explainability utilities."""

from nanonet.inspection.formatter import (
    format_bytes,
    format_execution_trace,
    format_inspection_report,
)
from nanonet.inspection.inspector import inspect_model, leaf_modules
from nanonet.inspection.report import (
    ActivationStats,
    GradientStats,
    LayerInspection,
    ModelInspectionReport,
    ModelTrace,
    TensorTraceInfo,
    TraceStep,
)
from nanonet.inspection.tracer import trace_model

__all__ = [
    "ActivationStats",
    "GradientStats",
    "LayerInspection",
    "ModelInspectionReport",
    "ModelTrace",
    "TensorTraceInfo",
    "TraceStep",
    "format_bytes",
    "format_execution_trace",
    "format_inspection_report",
    "inspect_model",
    "leaf_modules",
    "trace_model",
]
