"""Build a structured view of a Tensor's autograd computation graph."""

from __future__ import annotations

from nanonet.inspection.formatter import format_computation_graph
from nanonet.inspection.report import (
    ComputationGraph,
    GraphEdge,
    GraphOperationNode,
    GraphTensorNode,
)
from nanonet.inspection.utils import extract_tensor_metadata
from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor


def _topo_leaves_to_root(root: Tensor) -> list[Tensor]:
    """Return reachable tensors in leaves → root order (DFS post-order).

    Shared tensors appear once. Cycle-safe via a visiting set.
    """
    order: list[Tensor] = []
    done: set[int] = set()
    visiting: set[int] = set()

    def visit(node: Tensor) -> None:
        nid = id(node)
        if nid in done:
            return
        if nid in visiting:
            return
        visiting.add(nid)
        for parent in node._parents:
            visit(parent)
        visiting.discard(nid)
        done.add(nid)
        order.append(node)

    visit(root)
    return order


def _op_depths(root: Tensor, tensors: list[Tensor]) -> dict[int, int]:
    """Longest path length in *operations* from any leaf to each tensor."""
    depths: dict[int, int] = {}
    for node in tensors:  # already leaves → root
        nid = id(node)
        if node._grad_fn is None:
            depths[nid] = 0
        else:
            parent_depths = [depths.get(id(p), 0) for p in node._parents]
            depths[nid] = 1 + (max(parent_depths) if parent_depths else 0)
    return depths


def build_computation_graph(root: Tensor) -> ComputationGraph:
    """Collect a metadata-only DAG for the autograd history ending at ``root``."""
    tensors = _topo_leaves_to_root(root)
    depths = _op_depths(root, tensors)

    id_map: dict[int, str] = {}
    t_count = 0
    p_count = 0
    for node in tensors:
        nid = id(node)
        if isinstance(node, Parameter):
            label = f"P{p_count}"
            p_count += 1
        else:
            label = f"T{t_count}"
            t_count += 1
        id_map[nid] = label

    tensor_nodes: list[GraphTensorNode] = []
    operations: list[GraphOperationNode] = []
    edges: list[GraphEdge] = []
    op_count = 0

    for node in tensors:
        nid = id(node)
        label = id_map[nid]
        is_leaf = node._grad_fn is None
        is_param = isinstance(node, Parameter)
        meta = extract_tensor_metadata(node)
        tensor_nodes.append(
            GraphTensorNode(
                id=label,
                shape=meta["shape"],
                dtype=meta["dtype"],
                requires_grad=bool(meta["requires_grad"]),
                has_grad=bool(meta["has_grad"]),
                grad_shape=meta["grad_shape"],
                is_leaf=is_leaf,
                is_parameter=is_param,
                is_root=(nid == id(root)),
            )
        )

        if node._grad_fn is not None:
            op_id = f"OP{op_count}"
            op_count += 1
            op_name = node._grad_fn.op_name
            operations.append(GraphOperationNode(id=op_id, name=op_name))
            for parent in node._parents:
                edges.append(GraphEdge(source=id_map[id(parent)], target=op_id))
            edges.append(GraphEdge(source=op_id, target=label))

    root_id = id_map[id(root)]
    depth = depths.get(id(root), 0)
    leaf_count = sum(1 for n in tensor_nodes if n.is_leaf)
    parameter_count = sum(1 for n in tensor_nodes if n.is_parameter)

    return ComputationGraph(
        root_id=root_id,
        tensors=tensor_nodes,
        operations=operations,
        edges=edges,
        depth=depth,
        leaf_count=leaf_count,
        parameter_count=parameter_count,
    )


def inspect_computation_graph(
    root: Tensor,
    *,
    verbose: bool = True,
) -> ComputationGraph:
    """Build and optionally print the autograd graph rooted at ``root``.

    Does not call ``backward()``, mutate gradients, or modify parameters.
    """
    graph = build_computation_graph(root)
    if verbose:
        print(format_computation_graph(graph), end="")
    return graph
