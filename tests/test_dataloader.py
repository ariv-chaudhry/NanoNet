"""Tests for Dataset and DataLoader."""

import numpy as np

from nanonet.data import DataLoader, TensorDataset


def test_tensor_dataset():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    ds = TensorDataset(X, y)
    assert len(ds) == 10
    x0, y0 = ds[0]
    assert np.array_equal(x0, X[0])
    assert y0 == 0


def test_dataloader_batches():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (4, 2)
    assert batches[-1][0].shape == (2, 2)


def test_dataloader_drop_last_and_shuffle():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True, drop_last=True, seed=42)
    batches = list(loader)
    assert len(batches) == 2
    # Deterministic with seed
    loader2 = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True, drop_last=True, seed=42)
    b2 = list(loader2)
    assert np.allclose(batches[0][0], b2[0][0])
