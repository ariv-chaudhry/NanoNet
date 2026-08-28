"""Synthetic log anomaly classification with NanoNet LogDataset.

This example demonstrates a file-backed ML workflow:

    .log file → LogDataset(parser) → DataLoader → NanoNet model → train/eval

It is intentionally lightweight and educational — not production intrusion
detection. Labels and features are synthetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import nanonet_ml as nn
from nanonet_ml.data import DataLoader, LogDataset
from nanonet_ml.metrics import accuracy

# Semantic mappings live in the example — NanoNet does not ship log encoders.
LEVEL_MAP = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
SERVICE_MAP = {"auth": 0.0, "api": 1.0, "database": 2.0}
NUM_FEATURES = 4

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_LOG = DATA_DIR / "server_train.log"
TEST_LOG = DATA_DIR / "server_test.log"


def parse_log_line(line: str) -> tuple[list[float], int]:
    """Convert one log record into numerical features and an anomaly label.

    Format:
        TIMESTAMP LEVEL SERVICE STATUS LATENCY LABEL

    The trailing LABEL is the supervision target and is *not* included in
    the feature vector (no label leakage).
    """
    parts = line.split()
    if len(parts) != 6:
        raise ValueError(f"expected 6 fields, got {len(parts)}: {line!r}")

    _timestamp, level, service, status, latency, label = parts
    features = [
        LEVEL_MAP[level],
        SERVICE_MAP[service],
        float(status) / 500.0,
        float(latency) / 1000.0,
    ]
    return features, int(label)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NanoNet on synthetic server-log anomaly labels.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return mean loss and accuracy over ``loader``."""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0
    with nn.no_grad():
        for x_batch, y_batch in loader:
            x = nn.Tensor(x_batch)
            logits = model(x)
            loss = loss_fn(logits, y_batch)
            n = len(x_batch)
            total_loss += float(loss.data) * n
            total_correct += accuracy(logits, y_batch) * n
            total_n += n
    if total_n == 0:
        return 0.0, 0.0
    return total_loss / total_n, total_correct / total_n


def main() -> None:
    args = parse_args()
    nn.manual_seed(args.seed)

    # LogDataset: file-backed indexing + user parser.
    # DataLoader: batching / shuffling / NumPy collation.
    train_dataset = LogDataset(
        TRAIN_LOG,
        parser=parse_log_line,
        encoding="utf-8",
        skip_blank_lines=True,
    )
    test_dataset = LogDataset(
        TEST_LOG,
        parser=parse_log_line,
        encoding="utf-8",
        skip_blank_lines=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Training NanoNet on parsed server logs (synthetic anomaly demo)")
    print(f"Train records: {len(train_dataset)}")
    print(f"Test records:  {len(test_dataset)}")
    print(f"Features:      {NUM_FEATURES}")
    print()

    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Trainer.fit expects array inputs; use a manual loop over DataLoader.
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x = nn.Tensor(x_batch)
            logits = model(x)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(x_batch)
            running += float(loss.data) * n
            seen += n

        if epoch % 10 == 0 or epoch == 1:
            train_loss = running / max(seen, 1)
            print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print()
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # A few held-out predictions for readability.
    print()
    print("Example predictions:")
    model.eval()
    shown = 0
    with nn.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(nn.Tensor(x_batch))
            preds = np.argmax(logits.data, axis=1)
            labels = np.asarray(y_batch).astype(np.int64).reshape(-1)
            for expected, predicted in zip(labels, preds, strict=True):
                print(f"  expected={int(expected)} predicted={int(predicted)}")
                shown += 1
                if shown >= 6:
                    return


if __name__ == "__main__":
    main()
