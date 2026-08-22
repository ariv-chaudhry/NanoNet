"""Model inspection and explainability utilities."""

from nanonet_ml.inspection.diagnostics import diagnose_model
from nanonet_ml.inspection.formatter import (
    format_bytes,
    format_computation_graph,
    format_diagnostics_report,
    format_execution_trace,
    format_inspection_report,
    format_percentage,
    format_scientific,
    format_shape,
)
from nanonet_ml.inspection.graph import build_computation_graph, inspect_computation_graph
from nanonet_ml.inspection.inspector import inspect_model, leaf_modules
from nanonet_ml.inspection.report import (
    ActivationStats,
    ComputationGraph,
    DiagnosticFinding,
    DiagnosticsReport,
    GradientStats,
    GraphEdge,
    GraphOperationNode,
    GraphTensorNode,
    LayerInspection,
    ModelInspectionReport,
    ModelTrace,
    RuntimeActivationRecord,
    TensorTraceInfo,
    TraceStep,
)
from nanonet_ml.inspection.thresholds import DEFAULT_THRESHOLDS, DiagnosticThresholds
from nanonet_ml.inspection.tracer import trace_model

__all__ = [
    "ActivationStats",
    "ComputationGraph",
    "DEFAULT_THRESHOLDS",
    "DiagnosticFinding",
    "DiagnosticThresholds",
    "DiagnosticsReport",
    "GradientStats",
    "GraphEdge",
    "GraphOperationNode",
    "GraphTensorNode",
    "LayerInspection",
    "ModelInspectionReport",
    "ModelTrace",
    "RuntimeActivationRecord",
    "TensorTraceInfo",
    "TraceStep",
    "build_computation_graph",
    "diagnose_model",
    "format_bytes",
    "format_computation_graph",
    "format_diagnostics_report",
    "format_execution_trace",
    "format_inspection_report",
    "format_percentage",
    "format_scientific",
    "format_shape",
    "inspect_computation_graph",
    "inspect_model",
    "leaf_modules",
    "trace_model",
]
