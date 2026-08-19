"""Demonstrate NanoNet model inspection."""

from __future__ import annotations

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss


def main() -> None:
    manual_seed(0)

    model = Sequential(
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
    )

    print("=== Structural inspection ===\n")
    model.inspect()

    x = Tensor(np.random.randn(32, 784))
    print("\n=== Runtime inspection with sample batch ===\n")
    model.inspect(x)

    y = Tensor(np.zeros((32, 10)))
    loss = MSELoss()(model(x), y)
    loss.backward()

    print("\n=== After backward (gradients available) ===\n")
    model.inspect(verbose=False)
    report = model.inspect(x, verbose=True)
    print(f"\nProgrammatic access: {report.total_parameters:,} parameters")


if __name__ == "__main__":
    main()
