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


@dataclass
class TensorTraceInfo:
    """Trace-local identity and metadata for a Tensor observed during tracing."""

    trace_id: str
    shape: tuple[int, ...] | None
    dtype: str | None
    requires_grad: bool | None
    object_id: int | None = None


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
    module_object_id: int | None = None


@dataclass
class ModelTrace:
    """Structured result of ``model.trace(x)``.

    ``output`` retains the forward-pass result so autograd can continue from it.
    Timing fields include instrumentation overhead and are for debugging only.
    """

    model_name: str
    model_type: str
    steps: list[TraceStep]
    inputs: list[TensorTraceInfo]
    outputs: list[TensorTraceInfo]
    forward_duration_seconds: float
    traced_duration_seconds: float
    output: Any = None

    def __str__(self) -> str:
        from nanonet.inspection.formatter import format_execution_trace

        return format_execution_trace(self)


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


@dataclass
class GraphOperationNode:
    """A differentiable Function that produced a tensor."""

    id: str
    name: str


@dataclass
class GraphEdge:
    """Directed edge ``source -> target`` (forward dependency direction)."""

    source: str
    target: str


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

    @property
    def tensor_nodes(self) -> list[GraphTensorNode]:
        return self.tensors
