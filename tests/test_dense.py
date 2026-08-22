"""Tests for Dense layer."""

import numpy as np

from nanonet_ml import Tensor, manual_seed
from nanonet_ml.gradcheck import gradcheck
from nanonet_ml.layers import Dense


def test_dense_shapes():
    layer = Dense(784, 128)
    x = Tensor(np.random.randn(32, 784))
    out = layer(x)
    assert out.shape == (32, 128)
    assert layer.weight.shape == (784, 128)
    assert layer.bias is not None and layer.bias.shape == (128,)


def test_dense_no_bias():
    layer = Dense(10, 5, bias=False)
    assert layer.bias is None
    out = layer(Tensor(np.ones((4, 10))))
    assert out.shape == (4, 5)


def test_dense_dim_mismatch():
    layer = Dense(784, 128)
    try:
        layer(Tensor(np.ones((8, 512))))
    except ValueError as exc:
        assert "784" in str(exc)
        assert "512" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_dense_parameters_and_backward():
    manual_seed(0)
    layer = Dense(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    out = layer(x).sum()
    out.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (3,)


def test_dense_gradcheck():
    manual_seed(1)
    layer = Dense(3, 2)

    def f(x):
        return layer(x).sum()

    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    # Check input gradient path through Dense weights as well via param grads
    result = gradcheck(f, [x], epsilon=1e-6)
    assert result.passed, f"max_abs={result.max_abs_error}, max_rel={result.max_rel_error}"
