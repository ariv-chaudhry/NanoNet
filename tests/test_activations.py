"""Tests for activation functions."""

import numpy as np

from nanonet_ml import Tensor
from nanonet_ml.gradcheck import gradcheck
from nanonet_ml.layers import ReLU, Sigmoid, Softmax, Tanh, relu, sigmoid
from nanonet_ml.layers.activations import softmax


def test_relu():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = relu(x)
    assert np.allclose(y.data, [0.0, 0.0, 2.0])
    y.sum().backward()
    assert np.allclose(x.grad, [0.0, 0.0, 1.0])


def test_sigmoid_stable():
    x = Tensor([1000.0, -1000.0, 0.0], requires_grad=True)
    y = sigmoid(x)
    assert np.allclose(y.data, [1.0, 0.0, 0.5], atol=1e-6)
    assert np.all(np.isfinite(y.data))


def test_tanh():
    layer = Tanh()
    x = Tensor([0.0], requires_grad=True)
    y = layer(x)
    assert np.isclose(y.data, 0.0)
    y.backward()
    assert np.isclose(x.grad, 1.0)


def test_softmax_sums_to_one():
    x = Tensor([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    y = softmax(x)
    assert np.allclose(y.data.sum(axis=1), [1.0, 1.0])
    assert np.all(np.isfinite(y.data))


def test_activation_modules():
    x = Tensor(np.random.randn(4, 5))
    for layer in (ReLU(), Sigmoid(), Tanh(), Softmax()):
        out = layer(x)
        assert out.shape == x.shape


def test_relu_gradcheck():
    x = Tensor(np.array([0.5, -0.3, 1.2]), requires_grad=True)
    result = gradcheck(lambda t: ReLU()(t).sum(), [x])
    assert result.passed
