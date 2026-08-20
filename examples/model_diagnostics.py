"""Demonstrate NanoNet model diagnostics."""

from __future__ import annotations

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss
from nanonet.nn import Module


def main() -> None:
    manual_seed(0)
    model = Sequential(
        Dense(4, 8),
        ReLU(),
        Dense(8, 2),
    )
    x = Tensor(np.random.randn(8, 4))

    print("=== Clean model (with input) ===\n")
    model.diagnose(x)

    print("=== After forward / loss / backward ===\n")
    y = Tensor(np.zeros((8, 2)))
    loss = MSELoss()(model(x), y)
    loss.backward()
    model.diagnose(x, verbose=False)
    report = model.diagnose(x, verbose=False)
    print(f"critical={report.critical} warnings={report.warnings}")
    print(f"codes={[f.code for f in report.findings if f.severity != 'info']}")

    print("\n=== Deliberate dead ReLU ===\n")

    class AlwaysNegative(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, t: Tensor) -> Tensor:
            return t * 0.0 - 1.0

    dead = Sequential(AlwaysNegative(), ReLU())
    dead.diagnose(Tensor(np.ones((4, 4))))


if __name__ == "__main__":
    main()
