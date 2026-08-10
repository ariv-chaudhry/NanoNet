"""Tests for Tensor operations and basic properties."""

import numpy as np

from nanonet import Tensor


def test_tensor_attributes():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
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
    # (2)*(2) + 16 = 20
    assert np.isclose(y.data, 20.0)
    y.backward()
    assert a.grad is not None and b.grad is not None


def test_negation():
    x = Tensor([1.0, -2.0], requires_grad=True)
    y = (-x).sum()
    y.backward()
    assert np.allclose(x.grad, [-1.0, -1.0])


def test_sum_mean():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    s = x.sum()
    assert np.isclose(s.data, 10.0)
    s.backward()
    assert np.allclose(x.grad, np.ones((2, 2)))

    x.zero_grad()
    m = x.mean()
    assert np.isclose(m.data, 2.5)
    m.backward()
    assert np.allclose(x.grad, np.full((2, 2), 0.25))


def test_reshape_transpose():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    y = x.reshape(3, 2).T
    assert y.shape == (2, 3)
    y.sum().backward()
    assert np.allclose(x.grad, np.ones((2, 3)))


def test_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a @ b
    assert np.allclose(c.data, [[19.0, 22.0], [43.0, 50.0]])
    c.sum().backward()
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_broadcasting_add():
    x = Tensor(np.ones((4, 3)), requires_grad=True)
    b = Tensor(np.arange(3.0), requires_grad=True)
    y = (x + b).sum()
    y.backward()
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, np.full(3, 4.0))


def test_exp_log():
    x = Tensor([0.5, 1.0, 2.0], requires_grad=True)
    y = x.exp().log().sum()
    assert np.allclose(y.data, x.data.sum())
    y.backward()
    assert np.allclose(x.grad, np.ones(3))


def test_indexing():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    y = x[0].sum()
    y.backward()
    assert np.allclose(x.grad, [[1, 1, 1], [0, 0, 0]])


def test_maximum():
    a = Tensor([-1.0, 2.0], requires_grad=True)
    b = Tensor([0.0, 1.0], requires_grad=True)
    y = a.maximum(b).sum()
    assert np.allclose(y.data, sum([0.0, 2.0]))
    y.backward()
    assert np.allclose(a.grad, [0.0, 1.0])
    assert np.allclose(b.grad, [1.0, 0.0])
