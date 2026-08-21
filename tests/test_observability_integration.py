"""Tests for cross-feature observability integration and API consistency."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss
from nanonet.nn import Module


def _model() -> Sequential:
    manual_seed(0)
    return Sequential(
        Dense(4, 8),
        ReLU(),
        Dense(8, 2),
    )


def test_full_observability_workflow():
    model = _model()
    x = Tensor(np.random.randn(4, 4))
    y = Tensor(np.zeros((4, 2)))

    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)

    pred = model(x)
    loss = MSELoss()(pred, y)
    graph = loss.graph(verbose=False)
    loss.backward()
    diagnostics = model.diagnose(x, verbose=False)

    assert inspection.runtime_captured
    assert inspection.total_parameters > 0
    assert len(trace.steps) == 3
    assert graph.depth >= 1
    assert diagnostics.gradients_available
    assert diagnostics.activations_analyzed


def test_naming_consistency_across_features():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    diagnostics = model.diagnose(x, verbose=False)

    inspect_names = [layer.name for layer in inspection.layers]
    trace_names = [step.module_name for step in trace.steps]
    assert inspect_names == ["0", "1", "2"]
    assert trace_names == ["0", "1", "2"]
    # Diagnose findings that target modules use the same hierarchical names.
    for finding in diagnostics.findings:
        if finding.target in {"0", "1", "2"}:
            assert finding.target in inspect_names


def test_shape_consistency_inspect_trace():
    model = _model()
    x = Tensor(np.ones((3, 4)))
    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    for layer, step in zip(inspection.layers, trace.steps, strict=True):
        assert layer.name == step.module_name
        assert layer.output_shape == step.outputs[0].shape


def test_parameter_count_consistency():
    model = _model()
    inspection = model.inspect(verbose=False)
    seen: set[int] = set()
    canonical = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        canonical += int(p.size)
    assert inspection.total_parameters == canonical


def test_gradient_and_parameter_preservation():
    model = _model()
    x = Tensor(np.random.randn(2, 4))
    y = Tensor(np.zeros((2, 2)))
    loss = MSELoss()(model(x), y)
    loss.backward()

    grads_before = {n: p.grad.copy() for n, p in model.named_parameters() if p.grad is not None}
    params_before = {n: p.data.copy() for n, p in model.named_parameters()}
    x_before = x.data.copy()

    model.inspect(x, verbose=False)
    model.trace(x, verbose=False)
    loss.graph(verbose=False)
    model.diagnose(x, verbose=False)

    for n, g in grads_before.items():
        assert np.allclose(dict(model.named_parameters())[n].grad, g)
    for n, data in params_before.items():
        assert np.allclose(dict(model.named_parameters())[n].data, data)
    assert np.allclose(x.data, x_before)


def test_shared_module_semantics():
    class Dual(Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = Dense(4, 4)
            self.relu = ReLU()

        def forward(self, x: Tensor) -> Tensor:
            return self.relu(self.shared(self.shared(x)))

    model = Dual()
    x = Tensor(np.ones((2, 4)))
    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    diagnostics = model.diagnose(x, verbose=False)

    # Parameter tensor counted once
    n_unique = len({id(p) for p in model.parameters()})
    assert n_unique == 2  # weight + bias
    seen: set[int] = set()
    canonical = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        canonical += int(p.size)
    assert inspection.total_parameters == canonical

    # Trace records multiple invocations of shared Dense
    shared_steps = [s for s in trace.steps if s.module_name == "shared"]
    assert len(shared_steps) == 2
    assert shared_steps[0].call_index == 1
    assert shared_steps[1].call_index == 2

    assert diagnostics.activations_analyzed


def test_nested_module_names():
    class Block(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = Dense(3, 2)

        def forward(self, x: Tensor) -> Tensor:
            return self.fc(x)

    class Net(Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Block()

        def forward(self, x: Tensor) -> Tensor:
            return self.encoder(x)

    model = Net()
    x = Tensor(np.ones((2, 3)))
    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    assert inspection.layers[0].name == "encoder.fc"
    assert trace.steps[0].module_name == "encoder.fc"


def test_failed_forward_recovery_all_apis():
    model = _model()
    bad = Tensor(np.ones((2, 3)))
    with pytest.raises(ValueError):
        model.inspect(bad, verbose=False)
    with pytest.raises(ValueError):
        model.trace(bad, verbose=False)
    with pytest.raises(ValueError):
        model.diagnose(bad, verbose=False)

    good = Tensor(np.ones((2, 4)))
    out = model(good)
    assert out.shape == (2, 2)


def test_repeated_observability_calls():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    for _ in range(3):
        model.inspect(x, verbose=False)
        model.trace(x, verbose=False)
        model.diagnose(x, verbose=False)
    out = model(x)
    assert out.shape == (2, 2)


def test_trace_output_graph():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    trace = model.trace(x, verbose=False)
    graph = trace.output.graph(verbose=False)
    assert graph.root_id
    assert len(graph.tensors) >= 1


def test_diagnose_before_and_after_backward():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    before = model.diagnose(verbose=False)
    assert "GRAD_UNAVAILABLE" in {f.code for f in before.findings}
    assert "GRAD_MISSING" not in {f.code for f in before.findings}

    loss = MSELoss()(model(x), Tensor(np.zeros((2, 2))))
    loss.backward()
    graph = loss.graph(verbose=False)
    after = model.diagnose(verbose=False)
    assert after.gradients_available
    assert any(n.has_grad for n in graph.tensors if n.is_parameter)


def test_to_dict_json_serializable():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    loss = MSELoss()(model(x), Tensor(np.zeros((2, 2))))
    graph = loss.graph(verbose=False)
    diagnostics = model.diagnose(x, verbose=False)

    for report in (inspection, trace, graph, diagnostics):
        data = report.to_dict()
        json.dumps(data)


def test_verbose_false_all_silent(capsys):
    model = _model()
    x = Tensor(np.ones((2, 4)))
    model.inspect(x, verbose=False)
    model.trace(x, verbose=False)
    model(x).graph(verbose=False)
    model.diagnose(x, verbose=False)
    assert capsys.readouterr().out == ""


def test_str_reports_nonempty():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    assert "NanoNet Model Inspector" in str(model.inspect(x, verbose=False))
    assert "NanoNet Execution Trace" in str(model.trace(x, verbose=False))
    assert "NanoNet Computation Graph" in str(model(x).graph(verbose=False))
    assert "NanoNet Diagnostics" in str(model.diagnose(x, verbose=False))


def test_sequential_feature_orders():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    model.inspect(x, verbose=False)
    model.trace(x, verbose=False)
    model.diagnose(x, verbose=False)
    assert model(x).shape == (2, 2)

    model.trace(x, verbose=False)
    model.inspect(x, verbose=False)
    model.diagnose(x, verbose=False)
    assert model(x).shape == (2, 2)


def test_dtype_formatting_consistent():
    model = _model()
    x = Tensor(np.ones((2, 4)))
    trace = model.trace(x, verbose=False)
    graph = model(x).graph(verbose=False)
    for step in trace.steps:
        for t in step.outputs:
            assert t.dtype in {"float64", "float32"}
    for node in graph.tensors:
        assert node.dtype in {"float64", "float32"}
