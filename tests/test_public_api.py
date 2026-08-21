"""Tests for the public ``import nanonet as nn`` package API."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import numpy as np

import nanonet as nn


def test_version():
    assert nn.__version__ == "0.1.0"
    assert isinstance(nn.__version__, str)


def test_public_symbols_present():
    for name in nn.__all__:
        assert hasattr(nn, name), name


def test_star_import_only_exports_all():
    namespace: dict = {}
    exec("from nanonet import *", namespace)  # noqa: S102
    public = {k for k in namespace if k in nn.__all__}
    assert public == set(nn.__all__)
    assert "inspect_model" not in namespace
    assert "format_shape" not in namespace
    assert "build_context" not in namespace


def test_import_is_silent(capsys):
    import importlib

    importlib.reload(nn)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_top_level_model_forward():
    nn.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = nn.Tensor(np.ones((3, 4)))
    y = model(x)
    assert y.shape == (3, 2)


def test_dense_and_linear_equivalent():
    nn.manual_seed(1)
    d = nn.Dense(3, 2)
    nn.manual_seed(1)
    lin = nn.Linear(3, 2)
    x = nn.Tensor(np.ones((2, 3)))
    assert np.allclose(d(x).data, lin(x).data)
    assert isinstance(lin, nn.Dense)
    assert type(lin).__name__ == "Linear"


def test_optimizer_and_loss_from_top_level():
    nn.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 1))
    opt = nn.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x = nn.Tensor([[1.0, 2.0]])
    y = nn.Tensor([[0.0]])
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert float(loss.data) >= 0.0


def test_observability_via_public_api():
    nn.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = nn.Tensor(np.ones((2, 4)))

    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    out = model(x)
    graph = out.graph(verbose=False)
    diagnostics = model.diagnose(x, verbose=False)

    assert inspection.total_parameters > 0
    assert len(trace.steps) == 3
    assert graph.root_id
    assert diagnostics.ok or diagnostics.critical == 0
    json.dumps(inspection.to_dict())
    json.dumps(trace.to_dict())
    json.dumps(graph.to_dict())
    json.dumps(diagnostics.to_dict())


def test_sgd_and_cross_entropy_available():
    assert callable(nn.SGD)
    assert callable(nn.CrossEntropyLoss)
    assert callable(nn.no_grad)


def test_public_api_print_silence():
    nn.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    x = nn.Tensor(np.ones((1, 2)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.inspect(verbose=False)
        model.trace(x, verbose=False)
        model.diagnose(x, verbose=False)
        model(x).graph(verbose=False)
    assert buf.getvalue() == ""
