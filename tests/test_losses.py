"""Tests for loss functions."""

import numpy as np

from nanonet import Tensor
from nanonet.losses import CrossEntropyLoss, MSELoss


def test_mse_known_value():
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = Tensor([[1.0, 2.0], [3.0, 5.0]])
    loss = MSELoss()(pred, target)
    # errors: 0,0,0,1 -> mean 0.25
    assert np.isclose(loss.data, 0.25)
    loss.backward()
    assert pred.grad is not None


def test_cross_entropy_manual():
    # Single example, two classes: logits [0, 0] -> uniform 0.5 -> -log(0.5)
    logits = Tensor([[0.0, 0.0]], requires_grad=True)
    targets = np.array([1])
    loss = CrossEntropyLoss()(logits, targets)
    expected = -np.log(0.5)
    assert np.isclose(loss.data, expected, rtol=1e-5)


def test_cross_entropy_extreme_logits():
    logits = Tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    targets = np.array([0, 1])
    loss = CrossEntropyLoss()(logits, targets)
    assert np.isfinite(loss.data)
    loss.backward()
    assert np.all(np.isfinite(logits.grad))


def test_cross_entropy_label_validation():
    logits = Tensor(np.zeros((3, 5)), requires_grad=True)
    try:
        CrossEntropyLoss()(logits, np.array([[1, 2, 3]]))
    except ValueError as exc:
        assert "integer labels" in str(exc)
    else:
        raise AssertionError("expected ValueError")
