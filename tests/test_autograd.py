"""Tests for reverse-mode automatic differentiation."""

import numpy as np

from nanonet_ml import Tensor
from nanonet_ml.utils import unbroadcast


def test_simple_derivative():
    x = Tensor(3.0, requires_grad=True)
    y = x**2 + 2 * x
    y.backward()
    assert np.isclose(x.grad, 8.0)


def test_branching_graph():
    x = Tensor(2.0, requires_grad=True)
    a = x * x
    b = x * 3
    y = a + b
    y.backward()
    assert np.isclose(x.grad, 7.0)


def test_reused_tensor_accumulation():
    x = Tensor(4.0, requires_grad=True)
    y = x + x + x
    y.backward()
    assert np.isclose(x.grad, 3.0)


def test_gradient_accumulation():
    x = Tensor(1.0, requires_grad=True)
    y = x * 2
    y.backward()
    assert np.isclose(x.grad, 2.0)

    z = x * 3
    z.backward()
    assert np.isclose(x.grad, 5.0)

    x.zero_grad()
    assert x.grad is None


def test_chained_derivative():
    x = Tensor(2.0, requires_grad=True)
    y = (x * 3 + 1) ** 2
    y.backward()

    # dy/dx = 2*(3x+1)*3 = 6*(6+1) = 42
    assert np.isclose(x.grad, 42.0)


def test_unbroadcast_helper():
    grad = np.ones((32, 128))
    reduced = unbroadcast(grad, (128,))
    assert reduced.shape == (128,)
    assert np.allclose(reduced, 32.0)

    reduced2 = unbroadcast(grad, (1, 128))
    assert reduced2.shape == (1, 128)


def test_dense_bias_broadcast_grad():
    x = Tensor(
        np.random.randn(32, 16),
        requires_grad=True,
    )
    w = Tensor(
        np.random.randn(16, 8),
        requires_grad=True,
    )
    b = Tensor(
        np.random.randn(8),
        requires_grad=True,
    )

    y = (x @ w + b).sum()
    y.backward()

    assert b.grad.shape == (8,)
    assert x.grad.shape == (32, 16)
    assert w.grad.shape == (16, 8)


def test_multivariable():
    x = Tensor(1.0, requires_grad=True)
    w = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    y = x * w + b
    y.backward()

    assert np.isclose(x.grad, 2.0)
    assert np.isclose(w.grad, 1.0)
    assert np.isclose(b.grad, 1.0)


def test_repeated_backward_same_graph_accumulates_once_per_call():
    x = Tensor(2.0, requires_grad=True)
    y = x * x

    y.backward()
    assert np.isclose(x.grad, 4.0)

    y.backward()
    assert np.isclose(x.grad, 8.0)

    x.zero_grad()
    y.backward()
    assert np.isclose(x.grad, 4.0)