"""Lightweight end-to-end smoke test for the log anomaly example pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import nanonet_ml as nn
from nanonet_ml.data import DataLoader, LogDataset

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TRAIN_LOG = EXAMPLES / "data" / "server_train.log"
TEST_LOG = EXAMPLES / "data" / "server_test.log"


def parse_log_line(line: str) -> tuple[list[float], int]:
    parts = line.split()
    assert len(parts) == 6
    _ts, level, service, status, latency, label = parts
    levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
    services = {"auth": 0.0, "api": 1.0, "database": 2.0}
    features = [
        levels[level],
        services[service],
        float(status) / 500.0,
        float(latency) / 1000.0,
    ]
    # Label is separate — not appended to features.
    return features, int(label)


def test_example_log_files_exist():
    assert TRAIN_LOG.is_file()
    assert TEST_LOG.is_file()


def test_log_example_pipeline_one_training_step():
    nn.manual_seed(0)

    dataset = LogDataset(
        TRAIN_LOG,
        parser=parse_log_line,
        skip_blank_lines=True,
    )
    assert len(dataset) > 0

    features, label = dataset[0]
    assert len(features) == 4
    assert label in (0, 1)
    assert all(isinstance(v, float) for v in features)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, seed=0)
    x_batch, y_batch = next(iter(loader))
    assert isinstance(x_batch, np.ndarray)
    assert isinstance(y_batch, np.ndarray)
    assert x_batch.shape[1] == 4
    assert y_batch.shape == (x_batch.shape[0],)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    optimizer = nn.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    x = nn.Tensor(x_batch)
    logits = model(x)
    loss = loss_fn(logits, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert np.isfinite(float(loss.data))
    assert logits.shape == (x_batch.shape[0], 2)


def test_log_example_test_file_parses():
    dataset = LogDataset(TEST_LOG, parser=parse_log_line, skip_blank_lines=True)
    assert len(dataset) >= 8
    for i in range(min(5, len(dataset))):
        features, label = dataset[i]
        assert len(features) == 4
        assert label in (0, 1)
