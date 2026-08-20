"""Model inspection and explainability utilities."""

from nanonet.inspection.formatter import (
    format_bytes,
    format_computation_graph,
    format_execution_trace,
    format_inspection_report,
)
from nanonet.inspection.graph import build_computation_graph, inspect_computation_graph
from nanonet.inspection.inspector import inspect_model, leaf_modules
from nanonet.inspection.report import (
    ActivationStats,
    ComputationGraph,
    GradientStats,
    GraphEdge,
    GraphOperationNode,
    GraphTensorNode,
    LayerInspection,
    ModelInspectionReport,
    ModelTrace,
    TensorTraceInfo,
    TraceStep,
)
from nanonet.inspection.tracer import trace_model

__all__ = [
    "ActivationStats",
    "ComputationGraph",
    "GradientStats",
    "GraphEdge",
    "GraphOperationNode",
    "GraphTensorNode",
    "LayerInspection",
    "ModelInspectionReport",
    "ModelTrace",
    "TensorTraceInfo",
    "TraceStep",
    "build_computation_graph",
    "format_bytes",
    "format_computation_graph",
    "format_execution_trace",
    "format_inspection_report",
    "inspect_computation_graph",
    "inspect_model",
    "leaf_modules",
    "trace_model",
]
