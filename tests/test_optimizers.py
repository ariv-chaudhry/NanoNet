"""Tests for SGD and Adam optimizers."""

import numpy as np

from nanonet_ml import Tensor, manual_seed
from nanonet_ml.layers import Dense
from nanonet_ml.nn import Parameter
from nanonet_ml.optimizers import SGD, Adam


def test_sgd_moves_against_gradient():
    p = Parameter(np.array([1.0, -1.0]))
    p.grad = np.array([0.5, -0.5])
    opt = SGD([p], lr=0.1)
    opt.step()
    assert np.allclose(p.data, [0.95, -0.95])


def test_sgd_momentum_and_weight_decay():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    opt = SGD([p], lr=0.1, momentum=0.9, weight_decay=0.1)
    opt.step()
    # grad = 1 + 0.1*1 = 1.1; v = 1.1; p = 1 - 0.1*1.1 = 0.89
    assert np.isclose(p.data, 0.89)


def test_adam_first_step_manual():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, -0.2])
    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    opt = Adam([p], lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    opt.step()

    g = np.array([0.1, -0.2])
    m = (1 - beta1) * g
    v = (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)
    expected = np.array([1.0, 2.0]) - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(p.data, expected)


def test_zero_grad():
    manual_seed(0)
    layer = Dense(4, 2)
    x = Tensor(np.ones((2, 4)))
    (layer(x).sum()).backward()
    assert layer.weight.grad is not None
    opt = SGD(layer.parameters(), lr=0.01)
    opt.zero_grad()
    assert layer.weight.grad is None
