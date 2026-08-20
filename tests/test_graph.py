"""Tests for ``tensor.graph()`` and the computation-graph subsystem."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.inspection import ComputationGraph
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss
from nanonet.nn import Parameter


def test_simple_mul_graph():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    graph = c.graph(verbose=False)
    assert isinstance(graph, ComputationGraph)
    assert graph.root_id in {n.id for n in graph.tensors}
    assert len(graph.tensors) == 3
    assert len(graph.operations) == 1
    assert graph.operations[0].name == "Mul"
    assert graph.leaf_count == 2
    assert graph.depth == 1
    assert graph.edges  # parents -> OP -> result


def test_chained_arithmetic_depth():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = a * b
    d = c + a
    e = d.sum()
    graph = e.graph(verbose=False)
    assert graph.depth == 3
    names = {op.name for op in graph.operations}
    assert names == {"Mul", "Add", "Sum"}
    root = next(n for n in graph.tensors if n.is_root)
    assert root.id == graph.root_id
    assert root.shape == ()


def test_shared_dependency_not_duplicated():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = a * 2
    c = b + b
    graph = c.graph(verbose=False)
    # Shared intermediate ``b`` must appear exactly once.
    assert len([n for n in graph.tensors if not n.is_leaf and not n.is_root]) >= 1
    add_ops = [op for op in graph.operations if op.name == "Add"]
    assert len(add_ops) == 1
    add_id = add_ops[0].id
    parents = [e.source for e in graph.edges if e.target == add_id]
    assert len(parents) == 2
    assert parents[0] == parents[1]
    # Unique tensor IDs
    ids = [n.id for n in graph.tensors]
    assert len(ids) == len(set(ids))


def test_leaf_graph():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    graph = x.graph(verbose=False)
    assert len(graph.tensors) == 1
    assert len(graph.operations) == 0
    assert len(graph.edges) == 0
    assert graph.depth == 0
    assert graph.tensors[0].is_leaf
    assert graph.tensors[0].is_root


def test_non_grad_tensor_graph():
    x = Tensor([1.0, 2.0], requires_grad=False)
    graph = x.graph(verbose=False)
    assert len(graph.tensors) == 1
    assert graph.tensors[0].requires_grad is False
    assert graph.operations == []


def test_matmul_op_name():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[0.5], [0.25]], requires_grad=True)
    y = x @ w
    graph = y.graph(verbose=False)
    assert any(op.name == "MatMul" for op in graph.operations)


def test_reduction_sum():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    s = x.sum()
    graph = s.graph(verbose=False)
    assert any(op.name == "Sum" for op in graph.operations)
    assert graph.depth == 1


def test_relu_activation():
    x = Tensor([-1.0, 2.0], requires_grad=True)
    y = ReLU()(x)
    graph = y.graph(verbose=False)
    assert any(op.name == "ReLU" for op in graph.operations)


def test_model_loss_graph_structure():
    manual_seed(0)
    model = Sequential(Dense(4, 8), ReLU(), Dense(8, 2))
    x = Tensor(np.random.randn(3, 4))
    y = Tensor(np.zeros((3, 2)))
    loss = MSELoss()(model(x), y)
    graph = loss.graph(verbose=False)
    assert graph.parameter_count >= 4  # two Dense layers × weight+bias
    assert graph.depth >= 3
    assert any(op.name == "MatMul" for op in graph.operations)
    assert any(op.name == "ReLU" for op in graph.operations)
    assert any(n.is_parameter for n in graph.tensors)
    # Input participates as a leaf (may or may not require grad)
    assert graph.leaf_count >= graph.parameter_count


def test_graph_before_backward():
    a = Tensor([1.0], requires_grad=True)
    b = (a * 2).sum()
    graph = b.graph(verbose=False)
    assert all(not n.has_grad for n in graph.tensors)


def test_graph_after_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    loss = (a * 3).sum()
    loss.backward()
    graph = loss.graph(verbose=False)
    # Graph retained after backward in NanoNet
    assert len(graph.operations) >= 1
    leaf = next(n for n in graph.tensors if n.is_leaf and n.id.startswith("T"))
    assert leaf.has_grad is True


def test_graph_does_not_mutate_gradients():
    a = Tensor([1.0, 2.0], requires_grad=True)
    loss = (a * a).sum()
    loss.backward()
    before = a.grad.copy()
    loss.graph(verbose=False)
    assert np.allclose(a.grad, before)


def test_graph_does_not_mutate_parameters():
    manual_seed(2)
    model = Sequential(Dense(3, 2))
    x = Tensor(np.ones((2, 3)))
    loss = MSELoss()(model(x), Tensor(np.zeros((2, 2))))
    w_before = model[0].weight.data.copy()
    loss.graph(verbose=False)
    assert np.allclose(model[0].weight.data, w_before)


def test_repeated_graph_stable():
    a = Tensor([1.0], requires_grad=True)
    b = (a * 2 + a).sum()
    g1 = b.graph(verbose=False)
    g2 = b.graph(verbose=False)
    g3 = b.graph(verbose=False)
    assert len(g1.tensors) == len(g2.tensors) == len(g3.tensors)
    assert len(g1.operations) == len(g2.operations) == len(g3.operations)
    assert len(g1.edges) == len(g2.edges) == len(g3.edges)
    assert [n.id for n in g1.tensors] == [n.id for n in g2.tensors]


def test_verbose_false_silent(capsys):
    a = Tensor([1.0], requires_grad=True)
    b = a * 2
    b.graph(verbose=False)
    assert capsys.readouterr().out == ""


def test_verbose_true_prints():
    a = Tensor([1.0], requires_grad=True)
    b = a * 2
    buf = io.StringIO()
    with redirect_stdout(buf):
        b.graph(verbose=True)
    text = buf.getvalue()
    assert "NanoNet Computation Graph" in text
    assert "Mul" in text
    assert "Graph Summary" in text
    assert "ROOT" in text


def test_str_graph():
    a = Tensor([1.0], requires_grad=True)
    graph = (a * 2).graph(verbose=False)
    text = str(graph)
    assert "NanoNet Computation Graph" in text


def test_parameter_ids():
    w = Parameter(np.ones((2, 3)))
    x = Tensor(np.ones((4, 2)), requires_grad=True)
    y = x @ w
    graph = y.graph(verbose=False)
    params = [n for n in graph.tensors if n.is_parameter]
    assert len(params) == 1
    assert params[0].id.startswith("P")


def test_deterministic_ids():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    g1 = (a * b).graph(verbose=False)
    g2 = (a * b).graph(verbose=False)
    # Same structure / ID scheme for equivalent independent graphs
    assert [n.id for n in g1.tensors] == [n.id for n in g2.tensors]
    assert [op.name for op in g1.operations] == [op.name for op in g2.operations]


def test_op_metadata_no_math_effect():
    """Operation labels must not change numerical results."""
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    out = (a * b + a).sum()
    out.backward()
    g = a.grad.copy()
    # Rebuild identical math
    a2 = Tensor([2.0, 3.0], requires_grad=True)
    b2 = Tensor([4.0, 5.0], requires_grad=True)
    ((a2 * b2 + a2).sum()).backward()
    assert np.allclose(a2.grad, g)
    assert a2._grad_fn is None  # leaf
    assert (a2 * b2)._grad_fn.op_name == "Mul"
