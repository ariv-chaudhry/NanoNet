"""Tests for numerical gradient checking."""

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.gradcheck import gradcheck
from nanonet.layers import Dense, ReLU


def test_gradcheck_mul():
    a = Tensor(np.array([1.5, -2.0]), requires_grad=True)
    b = Tensor(np.array([0.5, 3.0]), requires_grad=True)
    result = gradcheck(lambda x, y: (x * y).sum(), [a, b])
    assert result.passed


def test_gradcheck_matmul():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 2), requires_grad=True)
    result = gradcheck(lambda x, y: (x @ y).sum(), [a, b], epsilon=1e-6)
    assert result.passed, result


def test_gradcheck_small_network():
    manual_seed(0)
    model = Sequential([Dense(3, 4), ReLU(), Dense(4, 1)])

    def f(x):
        return model(x).sum()

    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    result = gradcheck(f, [x], epsilon=1e-6, atol=1e-4, rtol=1e-3)
    assert result.passed, f"abs={result.max_abs_error} rel={result.max_rel_error}"
