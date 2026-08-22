"""Train an MLP on MNIST with NanoNet."""

from __future__ import annotations

import argparse

import nanonet_ml as nn
from nanonet_ml.data import load_mnist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a NanoNet MLP on MNIST.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-limit", type=int, default=None, help="Optional subset of training samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nn.manual_seed(args.seed)

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    if args.train_limit is not None:
        X_train = X_train[: args.train_limit]
        y_train = y_train[: args.train_limit]

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    model.summary(input_shape=(784,))
    print(f"Trainable parameters: {model.num_parameters()}")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = model.fit(
        X_train,
        y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=True,
    )

    test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")
    print(f"Final train loss: {history.loss[-1]:.4f}")


if __name__ == "__main__":
    main()
