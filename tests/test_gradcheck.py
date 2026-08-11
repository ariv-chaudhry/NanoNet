"""Tests for numerical gradient checking."""

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.gradcheck import gradcheck
from nanonet.layers import Dense, ReLU


def test_gradcheck_mul():
    a = Tensor(
        np.array([1.5, -2.0]),
        requires_grad=True,
    )
    b = Tensor(
        np.array([0.5, 3.0]),
        requires_grad=True,
    )

    result = gradcheck(
        lambda x, y: (x * y).sum(),
        [a, b],
    )

    assert result.passed


def test_gradcheck_matmul():
    a = Tensor(
        np.random.randn(2, 3),
        requires_grad=True,
    )
    b = Tensor(
        np.random.randn(3, 2),
        requires_grad=True,
    )

    result = gradcheck(
        lambda x, y: (x @ y).sum(),
        [a, b],
        epsilon=1e-6,
    )

    assert result.passed, result


def test_gradcheck_small_network():
    manual_seed(0)

    model = Sequential(
        [
            Dense(3, 4),
            ReLU(),
            Dense(4, 1),
        ]
    )

    def f(x):
        return model(x).sum()

    x = Tensor(
        np.random.randn(2, 3),
        requires_grad=True,
    )

    result = gradcheck(
        f,
        [x],
        epsilon=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    assert result.passed, (
        f"abs={result.max_abs_error} "
        f"rel={result.max_rel_error}"
    )


def test_gradcheck_unused_input_treats_gradient_as_zero():
    x = Tensor(
        np.array([1.0, 2.0]),
        requires_grad=True,
    )
    y = Tensor(
        np.array([3.0, 4.0]),
        requires_grad=True,
    )

    result = gradcheck(
        lambda a, b: a.sum(),
        [x, y],
    )

    assert result.passed
    assert np.allclose(
        result.analytical[-2:],
        0.0,
    )
    assert np.allclose(
        result.numerical[-2:],
        0.0,
    )


def test_gradcheck_matmul_vector_and_batched_cases():
    vector_a = Tensor(
        np.array([0.5, -1.0, 2.0]),
        requires_grad=True,
    )
    vector_b = Tensor(
        np.array([1.5, 3.0, -0.5]),
        requires_grad=True,
    )

    assert gradcheck(
        lambda a, b: a @ b,
        [vector_a, vector_b],
        epsilon=1e-6,
    ).passed

    matrix = Tensor(
        np.random.randn(2, 3),
        requires_grad=True,
    )
    vector = Tensor(
        np.random.randn(3),
        requires_grad=True,
    )

    assert gradcheck(
        lambda a, b: (a @ b).sum(),
        [matrix, vector],
        epsilon=1e-6,
    ).passed

    vector = Tensor(
        np.random.randn(3),
        requires_grad=True,
    )
    matrix = Tensor(
        np.random.randn(3, 4),
        requires_grad=True,
    )

    assert gradcheck(
        lambda a, b: (a @ b).sum(),
        [vector, matrix],
        epsilon=1e-6,
    ).passed

    batch = Tensor(
        np.random.randn(2, 3, 4),
        requires_grad=True,
    )
    weights = Tensor(
        np.random.randn(4, 5),
        requires_grad=True,
    )

    assert gradcheck(
        lambda a, b: (a @ b).sum(),
        [batch, weights],
        epsilon=1e-6,
    ).passed