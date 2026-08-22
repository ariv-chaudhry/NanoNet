"""Tests for ``model.trace(x)`` execution tracing."""

from __future__ import annotations

import numpy as np

from nanonet_ml import Sequential, Tensor, manual_seed
from nanonet_ml.inspection import ModelTrace
from nanonet_ml.layers import Dense, ReLU
from nanonet_ml.losses import MSELoss
from nanonet_ml.nn import Module


def test_basic_sequential_trace_order_and_shapes():
    manual_seed(0)
    model = Sequential(Dense(4, 8), ReLU(), Dense(8, 2))
    x = Tensor(np.ones((2, 4)))
    trace = model.trace(x, verbose=False)
    assert isinstance(trace, ModelTrace)
    assert len(trace.steps) == 3
    assert [s.module_type for s in trace.steps] == ["Dense", "ReLU", "Dense"]
    assert [s.module_name for s in trace.steps] == ["0", "1", "2"]
    assert trace.steps[0].inputs[0].shape == (2, 4)
    assert trace.steps[0].outputs[0].shape == (2, 8)
    assert trace.steps[1].inputs[0].shape == (2, 8)
    assert trace.steps[2].outputs[0].shape == (2, 2)
    assert trace.output is not None
    assert trace.output.shape == (2, 2)


def test_tensor_id_propagation():
    model = Sequential(Dense(3, 5), ReLU(), Dense(5, 1))
    x = Tensor(np.random.randn(4, 3))
    trace = model.trace(x, verbose=False)
    assert trace.inputs[0].trace_id == "T0"
    # Step1 output should be step2 input
    assert trace.steps[0].outputs[0].trace_id == trace.steps[1].inputs[0].trace_id
    assert trace.steps[1].outputs[0].trace_id == trace.steps[2].inputs[0].trace_id
    assert trace.steps[0].outputs[0].object_id == trace.steps[1].inputs[0].object_id


def test_shared_module_two_steps():
    class SharedModel(Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = Dense(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            x = self.shared(x)
            return self.shared(x)

    model = SharedModel()
    x = Tensor(np.ones((2, 4)))
    trace = model.trace(x, verbose=False)
    assert len(trace.steps) == 2
    assert trace.steps[0].module_name == "shared"
    assert trace.steps[1].module_name == "shared"
    assert trace.steps[0].module_object_id == trace.steps[1].module_object_id
    assert trace.steps[0].call_index == 1
    assert trace.steps[1].call_index == 2


def test_nested_sequential_names():
    model = Sequential(
        Dense(4, 16),
        ReLU(),
        Sequential(Dense(16, 8), ReLU()),
        Dense(8, 2),
    )
    x = Tensor(np.random.randn(2, 4))
    trace = model.trace(x, verbose=False)
    assert [s.module_name for s in trace.steps] == ["0", "1", "2.0", "2.1", "3"]


def test_custom_module_runtime_order():
    class MLP(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc2 = Dense(8, 2)
            self.fc1 = Dense(4, 8)
            self.relu = ReLU()

        def forward(self, x: Tensor) -> Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    model = MLP()
    x = Tensor(np.ones((3, 4)))
    trace = model.trace(x, verbose=False)
    # Registration order is fc2, fc1, relu — execution order must differ.
    assert [s.module_name for s in trace.steps] == ["fc1", "relu", "fc2"]


def test_conditional_execution():
    class Conditional(Module):
        def __init__(self, use_a: bool) -> None:
            super().__init__()
            self.use_a = use_a
            self.a = Dense(4, 4)
            self.b = Dense(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            if self.use_a:
                return self.a(x)
            return self.b(x)

    x = Tensor(np.ones((2, 4)))
    ta = Conditional(True).trace(x, verbose=False)
    tb = Conditional(False).trace(x, verbose=False)
    assert len(ta.steps) == 1
    assert ta.steps[0].module_name == "a"
    assert len(tb.steps) == 1
    assert tb.steps[0].module_name == "b"


def test_timing_fields_non_negative():
    model = Sequential(Dense(4, 4), ReLU())
    trace = model.trace(Tensor(np.ones((2, 4))), verbose=False)
    assert trace.forward_duration_seconds >= 0
    assert trace.traced_duration_seconds >= 0
    for step in trace.steps:
        assert step.duration_seconds >= 0


def test_autograd_preserved_through_trace():
    manual_seed(0)
    model = Sequential(Dense(4, 5), ReLU(), Dense(5, 1))
    x = Tensor(np.random.randn(4, 4), requires_grad=False)
    trace = model.trace(x, verbose=False)
    loss = MSELoss()(trace.output, Tensor(np.zeros((4, 1))))
    loss.backward()
    assert model[0].weight.grad is not None


def test_gradients_and_params_unchanged_by_trace():
    manual_seed(1)
    model = Sequential(Dense(4, 3), ReLU(), Dense(3, 1))
    x = Tensor(np.random.randn(3, 4))
    loss = MSELoss()(model(x), Tensor(np.zeros((3, 1))))
    loss.backward()
    grads_before = {
        n: None if p.grad is None else p.grad.copy() for n, p in model.named_parameters()
    }
    state_before = model.state_dict()

    model.trace(x, verbose=False)

    for n, p in model.named_parameters():
        assert np.allclose(grads_before[n], p.grad)
    for k, arr in state_before.items():
        assert np.array_equal(arr, model.state_dict()[k])


def test_repeated_traces_no_instrumentation_leak():
    model = Sequential(Dense(4, 4), ReLU(), Dense(4, 2))
    x = Tensor(np.ones((2, 4)))
    for _ in range(3):
        t = model.trace(x, verbose=False)
        assert len(t.steps) == 3
    from nanonet_ml.inspection.inspector import leaf_modules

    for _n, mod in leaf_modules(model):
        assert "forward" not in mod.__dict__
    out = model(x)
    assert out.shape == (2, 2)


def test_failed_forward_cleans_instrumentation():
    model = Sequential(Dense(4, 2))
    try:
        model.trace(Tensor(np.ones((2, 8))), verbose=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    from nanonet_ml.inspection.inspector import leaf_modules

    for _n, mod in leaf_modules(model):
        assert "forward" not in mod.__dict__
    out = model(Tensor(np.ones((2, 4))))
    assert out.shape == (2, 2)


def test_verbose_false_silent(capsys):
    model = Sequential(Dense(4, 2))
    model.trace(Tensor(np.ones((2, 4))), verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_tuple_output_handled():
    class Pair(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = Dense(4, 4)

        def forward(self, x: Tensor):
            y = self.fc(x)
            return y, y

    class Wrapper(Module):
        def __init__(self) -> None:
            super().__init__()
            self.pair = Pair()

        def forward(self, x: Tensor):
            return self.pair(x)

    model = Wrapper()
    x = Tensor(np.ones((2, 4)))
    trace = model.trace(x, verbose=False)
    # Leaf is Dense inside Pair (Pair has child).
    assert any(s.module_type == "Dense" for s in trace.steps)
    assert len(trace.outputs) == 2


def test_str_trace():
    model = Sequential(Dense(4, 2))
    trace = model.trace(Tensor(np.ones((1, 4))), verbose=False)
    text = str(trace)
    assert "NanoNet Execution Trace" in text
    assert "Steps: 1" in text


def test_inspect_still_works():
    model = Sequential(Dense(4, 8), ReLU(), Dense(8, 2))
    report = model.inspect(verbose=False)
    assert report.total_parameters == (4 * 8 + 8) + (8 * 2 + 2)
    x = Tensor(np.ones((2, 4)))
    report2 = model.inspect(x, verbose=False)
    assert report2.runtime_captured
