"""Demonstrate NanoNet model inspection."""

from __future__ import annotations

import numpy as np

import nanonet as nn


def main() -> None:
    nn.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    print("=== Structural inspection ===\n")
    model.inspect()

    x = nn.Tensor(np.random.randn(32, 784))
    print("\n=== Runtime inspection with sample batch ===\n")
    model.inspect(x)

    y = nn.Tensor(np.zeros((32, 10)))
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    print("\n=== After backward (gradients available) ===\n")
    model.inspect(verbose=False)
    report = model.inspect(x, verbose=True)
    print(f"\nProgrammatic access: {report.total_parameters:,} parameters")


if __name__ == "__main__":
    main()
