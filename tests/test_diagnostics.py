"""Tests for ``model.diagnose()`` and the diagnostics subsystem."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from nanonet import Sequential, Tensor, manual_seed
from nanonet.inspection import DiagnosticsReport, DiagnosticThresholds
from nanonet.layers import Dense, ReLU, Sigmoid, Tanh
from nanonet.losses import MSELoss
from nanonet.nn import Module, Parameter


def _clean_mlp() -> Sequential:
    manual_seed(0)
    return Sequential(Dense(4, 8), ReLU(), Dense(8, 2))


def test_diagnose_clean_model_no_false_positives():
    model = _clean_mlp()
    x = Tensor(np.random.randn(8, 4))
    report = model.diagnose(x, verbose=False)
    assert isinstance(report, DiagnosticsReport)
    assert report.critical == 0
    codes = {f.code for f in report.findings if f.severity in {"critical", "warning"}}
    assert "RELU_DEAD" not in codes
    assert "PARAM_NAN" not in codes
    assert report.ok


def test_diagnose_without_input_skips_activations():
    model = _clean_mlp()
    report = model.diagnose(verbose=False)
    assert report.activations_analyzed is False
    assert "ACT_SKIPPED" in {f.code for f in report.findings}
    assert "GRAD_UNAVAILABLE" in {f.code for f in report.findings}


def test_param_nan():
    model = Dense(3, 2)
    model.weight.data[0, 0] = np.nan
    report = model.diagnose(verbose=False)
    assert "PARAM_NAN" in {f.code for f in report.findings}
    assert any(f.code == "PARAM_NAN" and f.severity == "critical" for f in report.findings)


def test_param_inf():
    model = Dense(3, 2)
    model.weight.data[0, 0] = np.inf
    report = model.diagnose(verbose=False)
    assert "PARAM_INF" in {f.code for f in report.findings}


def test_grad_nan():
    model = Dense(2, 2)
    model.weight.grad = np.full_like(model.weight.data, np.nan)
    report = model.diagnose(verbose=False)
    assert "GRAD_NAN" in {f.code for f in report.findings}


def test_grad_inf():
    model = Dense(2, 2)
    model.weight.grad = np.full_like(model.weight.data, np.inf)
    report = model.diagnose(verbose=False)
    assert "GRAD_INF" in {f.code for f in report.findings}


def test_grad_exploding():
    model = Dense(2, 2)
    model.weight.grad = np.full_like(model.weight.data, 1e4)
    report = model.diagnose(verbose=False)
    assert "GRAD_EXPLODING" in {f.code for f in report.findings}
    finding = next(f for f in report.findings if f.code == "GRAD_EXPLODING")
    assert finding.observed_value is not None
    assert finding.threshold is not None


def test_grad_vanishing():
    model = Dense(2, 2)
    model.weight.grad = np.full_like(model.weight.data, 1e-12)
    model.bias.grad = np.full_like(model.bias.data, 1e-12)
    report = model.diagnose(verbose=False)
    assert "GRAD_VANISHING" in {f.code for f in report.findings}


def test_grad_imbalance():
    model = Sequential(Dense(4, 4), Dense(4, 2))
    model[0].weight.grad = np.full_like(model[0].weight.data, 1e-6)
    model[0].bias.grad = np.zeros_like(model[0].bias.data)
    model[1].weight.grad = np.full_like(model[1].weight.data, 1.0)
    model[1].bias.grad = np.zeros_like(model[1].bias.data)
    report = model.diagnose(verbose=False)
    assert "GRAD_IMBALANCE" in {f.code for f in report.findings}
    finding = next(f for f in report.findings if f.code == "GRAD_IMBALANCE")
    assert finding.observed_value is not None
    assert finding.observed_value >= 100.0


def test_grad_missing():
    class WithExtra(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = Dense(3, 2)
            self.orphan = Parameter(np.ones((2, 2)))

        def forward(self, x: Tensor) -> Tensor:
            return self.fc(x)

    model = WithExtra()
    x = Tensor(np.ones((2, 3)))
    y = Tensor(np.zeros((2, 2)))
    loss = MSELoss()(model(x), y)
    loss.backward()
    report = model.diagnose(verbose=False)
    assert "GRAD_MISSING" in {f.code for f in report.findings}


def test_no_false_grad_warnings_before_backward():
    model = _clean_mlp()
    report = model.diagnose(verbose=False)
    codes = {f.code for f in report.findings if f.severity == "warning"}
    assert "GRAD_VANISHING" not in codes
    assert "GRAD_EXPLODING" not in codes
    assert "GRAD_IMBALANCE" not in codes
    assert "GRAD_MISSING" not in codes


def test_dead_relu():
    class AlwaysNegative(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x * 0.0 - 1.0

    model = Sequential(AlwaysNegative(), ReLU())
    x = Tensor(np.ones((4, 4)))
    report = model.diagnose(x, verbose=False)
    assert "RELU_DEAD" in {f.code for f in report.findings}
    finding = next(f for f in report.findings if f.code == "RELU_DEAD")
    assert finding.observed_value is not None
    assert finding.observed_value >= 0.95


def test_normal_relu_not_flagged():
    manual_seed(1)
    model = Sequential(Dense(4, 8), ReLU())
    x = Tensor(np.linspace(-2, 2, 32).reshape(8, 4))
    report = model.diagnose(x, verbose=False)
    assert "RELU_DEAD" not in {f.code for f in report.findings}


def test_activation_nan_first_occurrence():
    class NanLayer(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return Tensor(np.full(x.shape, np.nan))

    class PassThrough(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x

    model = Sequential(NanLayer(), PassThrough())
    x = Tensor(np.ones((2, 3)))
    report = model.diagnose(x, verbose=False)
    nan_findings = [f for f in report.findings if f.code == "ACT_NAN"]
    assert len(nan_findings) == 1
    assert nan_findings[0].severity == "critical"
    assert nan_findings[0].target == "0"


def test_activation_inf():
    class InfLayer(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return Tensor(np.full(x.shape, np.inf))

    model = Sequential(InfLayer())
    report = model.diagnose(Tensor(np.ones((2, 2))), verbose=False)
    assert "ACT_INF" in {f.code for f in report.findings}


def test_sigmoid_saturation():
    class Huge(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x * 100.0

    model = Sequential(Huge(), Sigmoid())
    report = model.diagnose(Tensor(np.ones((4, 4))), verbose=False)
    assert "ACT_SATURATION" in {f.code for f in report.findings}


def test_sigmoid_not_saturated():
    model = Sequential(Sigmoid())
    report = model.diagnose(Tensor(np.linspace(-1, 1, 16).reshape(4, 4)), verbose=False)
    assert "ACT_SATURATION" not in {f.code for f in report.findings}


def test_tanh_saturation():
    class Huge(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x * 100.0

    model = Sequential(Huge(), Tanh())
    report = model.diagnose(Tensor(np.ones((4, 4))), verbose=False)
    assert "ACT_SATURATION" in {f.code for f in report.findings}


def test_low_variance_activation():
    class ConstantOut(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return Tensor(np.ones(x.shape) * 3.0)

    model = Sequential(ConstantOut())
    report = model.diagnose(Tensor(np.random.randn(4, 4)), verbose=False)
    assert "ACT_CONSTANT" in {f.code for f in report.findings}


def test_scalar_activation_not_constant_flagged():
    class ScalarOut(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x.sum()

    model = Sequential(ScalarOut())
    report = model.diagnose(Tensor(np.ones((2, 2))), verbose=False)
    assert "ACT_CONSTANT" not in {f.code for f in report.findings}


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
    model.encoder.fc.weight.data[0, 0] = np.nan
    report = model.diagnose(verbose=False)
    finding = next(f for f in report.findings if f.code == "PARAM_NAN")
    assert "encoder" in (finding.target or "")


def test_shared_module_invocations_distinct():
    class Dual(Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = ReLU()

        def forward(self, x: Tensor) -> Tensor:
            return self.shared(self.shared(x * 0.0 - 1.0))

    model = Dual()
    report = model.diagnose(Tensor(np.ones((4, 4))), verbose=False)
    dead = [f for f in report.findings if f.code == "RELU_DEAD"]
    assert len(dead) >= 1


def test_no_side_effects():
    model = _clean_mlp()
    x = Tensor(np.random.randn(4, 4))
    y = Tensor(np.zeros((4, 2)))
    loss = MSELoss()(model(x), y)
    loss.backward()
    w_before = model[0].weight.data.copy()
    g_before = model[0].weight.grad.copy()
    x_before = x.data.copy()
    model.diagnose(x, verbose=False)
    assert np.allclose(model[0].weight.data, w_before)
    assert np.allclose(model[0].weight.grad, g_before)
    assert np.allclose(x.data, x_before)


def test_repeated_diagnose_stable():
    model = _clean_mlp()
    x = Tensor(np.random.randn(4, 4))
    r1 = model.diagnose(x, verbose=False)
    r2 = model.diagnose(x, verbose=False)
    r3 = model.diagnose(x, verbose=False)
    assert [f.code for f in r1.findings] == [f.code for f in r2.findings]
    assert [f.code for f in r2.findings] == [f.code for f in r3.findings]
    out = model(x)
    assert out.shape == (4, 2)


def test_failed_forward_cleans_up():
    model = _clean_mlp()
    with pytest.raises(ValueError):
        model.diagnose(Tensor(np.ones((2, 3))), verbose=False)
    report = model.diagnose(Tensor(np.ones((2, 4))), verbose=False)
    assert isinstance(report, DiagnosticsReport)


def test_verbose_false_silent(capsys):
    model = _clean_mlp()
    model.diagnose(Tensor(np.ones((2, 4))), verbose=False)
    assert capsys.readouterr().out == ""


def test_verbose_true_prints():
    model = _clean_mlp()
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.diagnose(Tensor(np.ones((2, 4))), verbose=True)
    text = buf.getvalue()
    assert "NanoNet Diagnostics" in text


def test_str_report():
    model = _clean_mlp()
    report = model.diagnose(verbose=False)
    assert "NanoNet Diagnostics" in str(report)


def test_custom_thresholds():
    model = Dense(2, 2)
    model.weight.grad = np.full_like(model.weight.data, 0.5)
    thr = DiagnosticThresholds(exploding_gradient_norm=0.1)
    report = model.diagnose(verbose=False, thresholds=thr)
    assert "GRAD_EXPLODING" in {f.code for f in report.findings}


def test_finding_ordering():
    model = Dense(2, 2)
    model.weight.data[0, 0] = np.nan
    model.weight.grad = np.zeros_like(model.weight.data)
    model.bias.grad = np.full_like(model.bias.data, 1e-12)
    report = model.diagnose(verbose=False)
    severities = [f.severity for f in report.findings if f.severity != "info"]
    if len(severities) >= 2:
        assert severities == sorted(
            severities,
            key=lambda s: {"critical": 0, "warning": 1}.get(s, 9),
        )
