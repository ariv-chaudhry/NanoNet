"""Tests for Tensor operations and basic properties."""

import numpy as np

from nanonet_ml import Tensor


def test_tensor_attributes():
    t = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    assert t.shape == (2, 2)
    assert t.ndim == 2
    assert t.requires_grad is True
    assert t.grad is None


def test_addition_forward_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = a + b

    assert np.allclose(c.data, [4.0, 6.0])

    c.sum().backward()

    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])


def test_sub_mul_div_pow():
    a = Tensor(4.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    y = (a - b) * (a / b) + a**2

    assert np.isclose(y.data, 20.0)

    y.backward()

    assert a.grad is not None
    assert b.grad is not None


def test_negation():
    x = Tensor([1.0, -2.0], requires_grad=True)
    y = (-x).sum()
    y.backward()

    assert np.allclose(x.grad, [-1.0, -1.0])


def test_sum_mean():
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )

    s = x.sum()
    assert np.isclose(s.data, 10.0)

    s.backward()
    assert np.allclose(
        x.grad,
        np.ones((2, 2)),
    )

    x.zero_grad()

    m = x.mean()
    assert np.isclose(m.data, 2.5)

    m.backward()
    assert np.allclose(
        x.grad,
        np.full((2, 2), 0.25),
    )


def test_reshape_transpose():
    x = Tensor(
        np.arange(6.0).reshape(2, 3),
        requires_grad=True,
    )
    y = x.reshape(3, 2).T

    assert y.shape == (2, 3)

    y.sum().backward()

    assert np.allclose(
        x.grad,
        np.ones((2, 3)),
    )


def test_matmul():
    a = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    b = Tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        requires_grad=True,
    )

    c = a @ b

    assert np.allclose(
        c.data,
        [[19.0, 22.0], [43.0, 50.0]],
    )

    c.sum().backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_broadcasting_add():
    x = Tensor(
        np.ones((4, 3)),
        requires_grad=True,
    )
    b = Tensor(
        np.arange(3.0),
        requires_grad=True,
    )

    y = (x + b).sum()
    y.backward()

    assert b.grad.shape == (3,)
    assert np.allclose(
        b.grad,
        np.full(3, 4.0),
    )


def test_exp_log():
    x = Tensor(
        [0.5, 1.0, 2.0],
        requires_grad=True,
    )
    y = x.exp().log().sum()

    assert np.allclose(
        y.data,
        x.data.sum(),
    )

    y.backward()

    assert np.allclose(
        x.grad,
        np.ones(3),
    )


def test_indexing():
    x = Tensor(
        np.arange(6.0).reshape(2, 3),
        requires_grad=True,
    )
    y = x[0].sum()
    y.backward()

    assert np.allclose(
        x.grad,
        [[1, 1, 1], [0, 0, 0]],
    )


def test_maximum():
    a = Tensor(
        [-1.0, 2.0],
        requires_grad=True,
    )
    b = Tensor(
        [0.0, 1.0],
        requires_grad=True,
    )

    y = a.maximum(b).sum()

    assert np.allclose(
        y.data,
        sum([0.0, 2.0]),
    )

    y.backward()

    assert np.allclose(
        a.grad,
        [0.0, 1.0],
    )
    assert np.allclose(
        b.grad,
        [1.0, 0.0],
    )


def test_matmul_vector_vector_backward():
    a = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )
    b = Tensor(
        np.array([4.0, 5.0, 6.0]),
        requires_grad=True,
    )

    y = a @ b
    y.backward()

    assert y.shape == ()
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


def test_matmul_matrix_vector_backward():
    a = Tensor(
        np.arange(6.0).reshape(2, 3),
        requires_grad=True,
    )
    b = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )

    y = (a @ b).sum()
    y.backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert np.allclose(
        a.grad,
        np.broadcast_to(b.data, a.shape),
    )
    assert np.allclose(
        b.grad,
        a.data.sum(axis=0),
    )


def test_matmul_vector_matrix_backward():
    a = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )
    b = Tensor(
        np.arange(12.0).reshape(3, 4),
        requires_grad=True,
    )

    y = (a @ b).sum()
    y.backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert np.allclose(
        a.grad,
        b.data.sum(axis=1),
    )
    assert np.allclose(
        b.grad,
        np.broadcast_to(
            a.data[:, None],
            b.shape,
        ),
    )


def test_matmul_batched_broadcast_backward():
    a = Tensor(
        np.arange(24.0).reshape(2, 3, 4),
        requires_grad=True,
    )
    b = Tensor(
        np.arange(20.0).reshape(4, 5),
        requires_grad=True,
    )

    y = (a @ b).sum()
    y.backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_no_grad_disables_graph_construction_and_restores_state():
    from nanonet_ml import no_grad

    x = Tensor(2.0, requires_grad=True)

    with no_grad():
        y = x * x + 1

    assert y.requires_grad is False
    assert y._grad_fn is None

    z = x * x

    assert z.requires_grad is True
    assert z._grad_fn is not None


def test_detach_returns_independent_copy():
    x = Tensor(
        [1.0, 2.0],
        requires_grad=True,
    )
    y = x.detach()

    y.data[0] = 99.0

    assert x.data[0] == 1.0
    assert y.requires_grad is False