"""PyTorch-gated numerical parity tests (skipped if torch is unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.numerical_parity import run_parity  # noqa: E402
from benchmarks.utils import (  # noqa: E402
    assign_numpy_params_to_nanonet,
    compare_arrays,
    copy_nanonet_weights_to_pytorch,
    nanonet_mlp,
    pytorch_mlp,
)


def test_weight_layout_roundtrip():
    nn_model = nanonet_mlp((4, 5, 3))
    rng = np.random.default_rng(1)
    params = [
        (rng.normal(size=(4, 5)), rng.normal(size=(5,))),
        (rng.normal(size=(5, 3)), rng.normal(size=(3,))),
    ]
    assign_numpy_params_to_nanonet(nn_model, params)
    pt = pytorch_mlp((4, 5, 3), dtype=torch.float64)
    copy_nanonet_weights_to_pytorch(nn_model, pt)
    # PyTorch weight should be transpose of NanoNet weight
    w_nn = nn_model[0].weight.data
    w_pt = pt[0].weight.detach().numpy()
    assert np.allclose(w_nn, w_pt.T)


def test_numerical_parity_end_to_end():
    result = run_parity(seed=0, out=None)
    # Avoid writing during tests if out=None still writes — run_parity always saves.
    # Check core assertions.
    r = result["results"]
    assert r["forward"]["allclose"]
    assert r["loss"]["allclose"]
    assert r["gradients"]["allclose"]
    assert r["sgd_update"]["allclose"]
    assert r["overall_pass"]
    assert r["forward"]["max_abs_error"] < 1e-10
    assert r["gradients"]["overall_max_abs_error"] < 1e-10
    assert r["sgd_update"]["overall_max_abs_error"] < 1e-10


def test_compare_arrays_detects_mismatch():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.1])
    row = compare_arrays(a, b, rtol=1e-7, atol=1e-9, name="x")
    assert row["allclose"] is False
    assert row["max_abs_error"] == pytest.approx(0.1)
