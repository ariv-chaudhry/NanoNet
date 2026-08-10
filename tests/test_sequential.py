"""Tests for Sequential models."""

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, Dropout, Flatten, ReLU


def test_sequential_forward():
    model = Sequential([Dense(10, 8), ReLU(), Dense(8, 3)])
    x = Tensor(np.random.randn(5, 10))
    out = model(x)
    assert out.shape == (5, 3)


def test_sequential_varargs():
    model = Sequential(Dense(4, 4), ReLU(), Dense(4, 2))
    assert len(model) == 3
    assert model.num_parameters() > 0


def test_sequential_train_eval_dropout():
    manual_seed(0)
    model = Sequential([Dense(8, 8), Dropout(0.5)])
    x = Tensor(np.ones((16, 8)))
    model.train()
    y_train = model(x)
    model.eval()
    out1 = model(x)
    out2 = model(x)
    assert y_train.shape == out1.shape
    # Eval mode: Dropout is a deterministic pass-through
    assert np.allclose(out1.data, out2.data)


def test_parameters_recursive():
    model = Sequential([Dense(5, 4), ReLU(), Dense(4, 2)])
    params = model.parameters()
    assert len(params) == 4  # 2 weights + 2 biases


def test_summary_and_flatten():
    model = Sequential([Flatten(), Dense(784, 10)])
    x = Tensor(np.zeros((2, 28, 28)))
    out = model(x)
    assert out.shape == (2, 10)
    text = model.summary(input_shape=(28, 28))
    assert "Total parameters" in text
