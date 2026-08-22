"""Tests for the training loop."""

import numpy as np

from nanonet_ml import Sequential, manual_seed
from nanonet_ml.layers import Dense, ReLU
from nanonet_ml.losses import CrossEntropyLoss, MSELoss
from nanonet_ml.optimizers import SGD, Adam


def test_regression_fit_decreases_loss():
    manual_seed(0)
    rng = np.random.default_rng(0)

    X = rng.uniform(
        -1,
        1,
        size=(64, 1),
    )
    y = 3 * X**2 + 2 * X

    model = Sequential(
        [
            Dense(1, 16),
            ReLU(),
            Dense(16, 1),
        ]
    )

    opt = Adam(
        model.parameters(),
        lr=0.05,
    )
    loss_fn = MSELoss()

    history = model.fit(
        X,
        y,
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=20,
        batch_size=16,
        verbose=False,
    )

    assert history.loss[-1] < history.loss[0]


def test_classification_fit_and_evaluate():
    manual_seed(1)
    rng = np.random.default_rng(1)

    X = rng.normal(
        size=(100, 4),
    )
    y = (
        X[:, 0] + X[:, 1] > 0
    ).astype(np.int64)

    model = Sequential(
        [
            Dense(4, 8),
            ReLU(),
            Dense(8, 2),
        ]
    )

    opt = SGD(
        model.parameters(),
        lr=0.1,
    )
    loss_fn = CrossEntropyLoss()

    history = model.fit(
        X,
        y,
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=15,
        batch_size=20,
        validation_data=(
            X[:20],
            y[:20],
        ),
        verbose=False,
    )

    assert len(history.loss) == 15

    acc = model.evaluate(X, y)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_disables_autograd_graph_construction():
    class TrackingDense(Dense):
        def forward(self, x):
            out = super().forward(x)
            self.last_requires_grad = (
                out.requires_grad
            )
            return out

    rng = np.random.default_rng(7)
    X = rng.normal(size=(8, 3))
    y = rng.integers(
        0,
        2,
        size=8,
    )

    output = TrackingDense(3, 2)
    model = Sequential([output])

    model.evaluate(X, y)

    assert output.last_requires_grad is False