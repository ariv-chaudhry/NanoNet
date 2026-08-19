"""Tests for ``model.inspect()`` and the inspection subsystem."""

from __future__ import annotations

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.inspection import ModelInspectionReport, format_inspection_report
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss
from nanonet.nn import Module


def _mlp() -> Sequential:
    return Sequential(
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
    )


def test_inspect_basic_structure():
    model = _mlp()
    report = model.inspect(verbose=False)
    assert isinstance(report, ModelInspectionReport)
    assert report.model_type == "Sequential"
    assert report.layer_count == 3
    assert report.layers[0].type == "Dense"
    assert report.layers[1].type == "ReLU"
    assert report.layers[2].type == "Dense"
    expected = 784 * 128 + 128 + 128 * 10 + 10
    assert report.total_parameters == expected
    assert report.trainable_parameters == expected
    assert report.total_parameters == sum(int(p.size) for p in model.parameters())


def test_parameter_count_dense():
    layer = Dense(4, 8)
    report = layer.inspect(verbose=False)
    assert report.total_parameters == 4 * 8 + 8
    assert report.layer_count == 1
    assert report.layers[0].parameter_count == 40
    assert report.estimated_parameter_memory_bytes == sum(
        p.data.nbytes for p in layer.parameters()
    )


def test_inspect_empty_sequential():
    model = Sequential()
    report = model.inspect(verbose=False)
    assert report.total_parameters == 0
    text = format_inspection_report(report)
    assert "Parameters: 0" in text


def test_runtime_shapes_and_activations():
    manual_seed(0)
    model = Sequential(Dense(4, 8), ReLU(), Dense(8, 2))
    x = Tensor(np.ones((32, 4)))
    report = model.inspect(x, verbose=False)
    assert report.runtime_captured
    assert report.input_shape == (32, 4)
    assert report.output_shape == (32, 2)
    assert report.layers[0].input_shape == (32, 4)
    assert report.layers[0].output_shape == (32, 8)
    assert report.layers[1].input_shape == (32, 8)
    assert report.layers[1].output_shape == (32, 8)
    assert report.layers[2].output_shape == (32, 2)
    # ReLU zeros some activations on random init with ones input — check available.
    assert report.layers[0].activation.available
    assert report.layers[1].activation.available
    assert report.layers[1].activation.min == 0.0
    assert 0.0 <= report.layers[1].activation.zero_fraction <= 1.0


def test_gradients_before_and_after_backward():
    manual_seed(1)
    model = Sequential(Dense(4, 5), ReLU(), Dense(5, 1))
    report = model.inspect(verbose=False)
    assert report.gradients_available is False

    x = Tensor(np.random.randn(8, 4))
    y = Tensor(np.random.randn(8, 1))
    loss = MSELoss()(model(x), y)
    loss.backward()

    grads_before = {
        name: None if p.grad is None else p.grad.copy()
        for name, p in model.named_parameters()
    }
    report2 = model.inspect(verbose=False)
    assert report2.gradients_available
    assert all(g.exists for g in report2.gradients if "weight" in g.name or "bias" in g.name)
    for name, g in model.named_parameters():
        assert np.allclose(grads_before[name], g.grad)


def test_no_side_effects_on_parameters_or_hooks():
    manual_seed(2)
    model = Sequential(Dense(4, 6), ReLU(), Dense(6, 3))
    state_before = model.state_dict()
    from nanonet.inspection.inspector import leaf_modules

    leaves = leaf_modules(model)
    assert "forward" not in leaves[0][1].__dict__

    x = Tensor(np.random.randn(4, 4))
    model.inspect(x, verbose=False)

    state_after = model.state_dict()
    for key in state_before:
        assert np.array_equal(state_before[key], state_after[key])
    for _name, mod in leaves:
        assert "forward" not in mod.__dict__


def test_nested_sequential_names_and_params():
    model = Sequential(
        Dense(4, 16),
        ReLU(),
        Sequential(
            Dense(16, 8),
            ReLU(),
        ),
        Dense(8, 2),
    )
    report = model.inspect(verbose=False)
    names = [layer.name for layer in report.layers]
    assert names == ["0", "1", "2.0", "2.1", "3"]
    expected = (4 * 16 + 16) + (16 * 8 + 8) + (8 * 2 + 2)
    assert report.total_parameters == expected

    x = Tensor(np.random.randn(2, 4))
    report2 = model.inspect(x, verbose=False)
    assert report2.layers[2].name == "2.0"
    assert report2.layers[2].input_shape == (2, 16)
    assert report2.layers[2].output_shape == (2, 8)


def test_custom_module():
    class MLP(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = Dense(4, 8)
            self.relu = ReLU()
            self.fc2 = Dense(8, 2)

        def forward(self, x: Tensor) -> Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    model = MLP()
    report = model.inspect(verbose=False)
    assert report.model_type == "MLP"
    assert [layer.name for layer in report.layers] == ["fc1", "relu", "fc2"]
    assert report.total_parameters == (4 * 8 + 8) + (8 * 2 + 2)

    x = Tensor(np.ones((3, 4)))
    report2 = model.inspect(x, verbose=False)
    assert report2.output_shape == (3, 2)
    assert report2.layers[0].output_shape == (3, 8)


def test_inspect_raises_on_bad_input():
    model = Sequential(Dense(4, 2))
    try:
        model.inspect(Tensor(np.ones((2, 8))), verbose=False)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "4" in str(exc)


def test_unsupported_output_stats_graceful():
    class Weird(Module):
        def forward(self, x: Tensor):
            return "not-a-tensor"

    model = Sequential(Weird())
    x = Tensor(np.ones((2, 3)))
    report = model.inspect(x, verbose=False)
    assert report.runtime_captured
    assert report.layers[0].activation.available is False
    assert report.output_shape is None
