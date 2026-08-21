"""Structured data types for NanoNet observability reports.

These dataclasses separate collection from presentation so inspection, tracing,
graph inspection, and diagnostics can share records without parsing text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _shape_list(shape: tuple[int, ...] | None) -> list[int] | None:
    if shape is None:
        return None
    return list(shape)


@dataclass
class ActivationStats:
    """Basic statistics over a tensor-valued layer output."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    zero_fraction: float | None = None  # in [0, 1]
    available: bool = False
    nan_count: int = 0
    inf_count: int = 0
    abs_max: float | None = None
    element_count: int = 0
    saturation_fraction: float | None = None  # set for Sigmoid/Tanh diagnostics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticFinding:
    """One evidence-backed diagnostic result."""

    severity: str  # "critical" | "warning" | "info"
    category: str  # "parameters" | "gradients" | "activations" | "general"
    code: str
    message: str
    target: str | None = None
    observed_value: float | None = None
    threshold: float | None = None
    explanation: str | None = None
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeActivationRecord:
    """One leaf-module output observed during a diagnostic forward pass."""

    name: str
    module_type: str
    call_index: int
    stats: ActivationStats
    display_name: str = ""  # includes call suffix when shared

    def __post_init__(self) -> None:
        if not self.display_name:
            if self.call_index > 1:
                self.display_name = f"{self.name} [call {self.call_index}]"
            else:
                self.display_name = self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module_type": self.module_type,
            "call_index": self.call_index,
            "display_name": self.display_name,
            "stats": self.stats.to_dict(),
        }


@dataclass
class DiagnosticsReport:
    """Structured result of ``model.diagnose()`` / ``model.diagnose(x)``."""

    model_name: str
    findings: list[DiagnosticFinding]
    checks_run: int
    warnings: int
    critical: int
    info: int
    activations_analyzed: bool = False
    gradients_available: bool = False

    def __str__(self) -> str:
        from nanonet.inspection.formatter import format_diagnostics_report

        return format_diagnostics_report(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible export (no Tensor / Module objects)."""
        return {
            "model_name": self.model_name,
            "findings": [f.to_dict() for f in self.findings],
            "checks_run": self.checks_run,
            "warnings": self.warnings,
            "critical": self.critical,
            "info": self.info,
            "activations_analyzed": self.activations_analyzed,
            "gradients_available": self.gradients_available,
            "ok": self.ok,
        }

    @property
    def has_critical(self) -> bool:
        return self.critical > 0

    @property
    def has_warnings(self) -> bool:
        return self.warnings > 0

    @property
    def ok(self) -> bool:
        return self.critical == 0 and self.warnings == 0


@dataclass
class LayerInspection:
    """Inspection record for a single (typically leaf) module."""

    name: str
    type: str
    parameter_count: int
    trainable_parameter_count: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    activation: ActivationStats = field(default_factory=ActivationStats)
    # Internal object identities for correlating inspect records (not in to_dict).
    input_tensor_id: int | None = field(default=None, repr=False)
    output_tensor_id: int | None = field(default=None, repr=False)
    parameter_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "input_shape": _shape_list(self.input_shape),
            "output_shape": _shape_list(self.output_shape),
            "activation": self.activation.to_dict(),
            "parameter_names": list(self.parameter_names),
        }


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
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        from nanonet.inspection.formatter import format_inspection_report

        return format_inspection_report(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible export (no Tensor / Module objects)."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "layers": [layer.to_dict() for layer in self.layers],
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "non_trainable_parameters": self.non_trainable_parameters,
            "estimated_parameter_memory_bytes": self.estimated_parameter_memory_bytes,
            "input_shape": _shape_list(self.input_shape),
            "output_shape": _shape_list(self.output_shape),
            "gradients": [g.to_dict() for g in self.gradients],
            "gradients_available": self.gradients_available,
            "runtime_captured": self.runtime_captured,
            "layer_count": self.layer_count,
        }

    @property
    def layer_count(self) -> int:
        return len(self.layers)


@dataclass
class TensorTraceInfo:
    """Trace-local identity and metadata for a Tensor observed during tracing."""

    trace_id: str
    shape: tuple[int, ...] | None
    dtype: str | None
    requires_grad: bool | None
    object_id: int | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "shape": _shape_list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
        }


@dataclass
class TraceStep:
    """One chronological module execution event during a forward pass."""

    index: int
    module_name: str
    module_type: str
    inputs: list[TensorTraceInfo]
    outputs: list[TensorTraceInfo]
    parameter_count: int
    duration_seconds: float
    call_index: int = 1  # invocation number for this module name within the trace
    module_object_id: int | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "module_name": self.module_name,
            "module_type": self.module_type,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "parameter_count": self.parameter_count,
            "duration_seconds": self.duration_seconds,
            "call_index": self.call_index,
        }


@dataclass
class ModelTrace:
    """Structured result of ``model.trace(x)``.

    ``output`` retains the forward-pass result so autograd can continue from it.
    Timing fields include instrumentation overhead and are for debugging only.
    ``to_dict()`` exports metadata only and omits the live ``output`` Tensor.
    """

    model_name: str
    model_type: str
    steps: list[TraceStep]
    inputs: list[TensorTraceInfo]
    outputs: list[TensorTraceInfo]
    forward_duration_seconds: float
    traced_duration_seconds: float
    output: Any = field(default=None, repr=False)

    def __str__(self) -> str:
        from nanonet.inspection.formatter import format_execution_trace

        return format_execution_trace(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible export. Omits live ``output`` Tensor."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "steps": [s.to_dict() for s in self.steps],
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "forward_duration_seconds": self.forward_duration_seconds,
            "traced_duration_seconds": self.traced_duration_seconds,
            "has_output": self.output is not None,
        }


@dataclass
class GraphTensorNode:
    """Metadata for a Tensor in a computation graph (no array data retained)."""

    id: str
    shape: tuple[int, ...] | None
    dtype: str | None
    requires_grad: bool
    has_grad: bool
    grad_shape: tuple[int, ...] | None
    is_leaf: bool
    is_parameter: bool
    is_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shape": _shape_list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "has_grad": self.has_grad,
            "grad_shape": _shape_list(self.grad_shape),
            "is_leaf": self.is_leaf,
            "is_parameter": self.is_parameter,
            "is_root": self.is_root,
        }


@dataclass
class GraphOperationNode:
    """A differentiable Function that produced a tensor."""

    id: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name}


@dataclass
class GraphEdge:
    """Directed edge ``source -> target`` (forward dependency direction)."""

    source: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target}


@dataclass
class ComputationGraph:
    """Structured autograd graph rooted at the tensor that called ``.graph()``.

    Ordering is leaves → root (topological). Depth is the longest path counted
    in *operations* from any leaf to the root (0 for a single-node leaf graph).
    Nodes store metadata only — not Tensor/array copies.
    """

    root_id: str
    tensors: list[GraphTensorNode]
    operations: list[GraphOperationNode]
    edges: list[GraphEdge]
    depth: int
    leaf_count: int
    parameter_count: int

    def __str__(self) -> str:
        from nanonet.inspection.formatter import format_computation_graph

        return format_computation_graph(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible export (metadata only)."""
        return {
            "root_id": self.root_id,
            "tensors": [t.to_dict() for t in self.tensors],
            "operations": [o.to_dict() for o in self.operations],
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
            "leaf_count": self.leaf_count,
            "parameter_count": self.parameter_count,
        }

    @property
    def tensor_nodes(self) -> list[GraphTensorNode]:
        return self.tensors
