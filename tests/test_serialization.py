"""Tests for model serialization."""

import numpy as np

from nanonet_ml import Sequential, Tensor, manual_seed
from nanonet_ml.layers import Dense, ReLU


def test_state_dict_roundtrip():
    manual_seed(0)

    model = Sequential(
        [
            Dense(4, 8),
            ReLU(),
            Dense(8, 2),
        ]
    )

    state = model.state_dict()

    model2 = Sequential(
        [
            Dense(4, 8),
            ReLU(),
            Dense(8, 2),
        ]
    )

    model2.load_state_dict(state)

    x = Tensor(
        np.random.randn(3, 4)
    )

    assert np.allclose(
        model(x).data,
        model2(x).data,
    )


def test_save_load_file(tmp_path):
    manual_seed(1)

    model = Sequential(
        [
            Dense(5, 3),
            ReLU(),
            Dense(3, 2),
        ]
    )

    x = Tensor(
        np.random.randn(4, 5)
    )

    before = model(x).data.copy()

    path = tmp_path / "model.npz"

    model.save(path)

    model2 = Sequential(
        [
            Dense(5, 3),
            ReLU(),
            Dense(3, 2),
        ]
    )

    model2.load(path)

    after = model2(x).data

    assert np.allclose(
        before,
        after,
    )


def test_save_load_normalizes_missing_npz_extension(tmp_path):
    manual_seed(2)

    model = Sequential(
        [
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
        ]
    )

    x = Tensor(
        np.random.randn(2, 3)
    )

    before = model(x).data.copy()

    path_without_suffix = (
        tmp_path / "model"
    )

    model.save(
        path_without_suffix
    )

    assert (
        tmp_path / "model.npz"
    ).exists()

    assert (
        tmp_path / "model.npz.meta.json"
    ).exists()

    restored = Sequential(
        [
            Dense(3, 4),
            ReLU(),
            Dense(4, 2),
        ]
    )

    restored.load(
        path_without_suffix
    )

    assert np.allclose(
        before,
        restored(x).data,
    )