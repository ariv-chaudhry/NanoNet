"""Demonstrate NanoNet model diagnostics."""

from __future__ import annotations

import numpy as np

import nanonet as nn


def main() -> None:
    nn.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = nn.Tensor(np.random.randn(8, 4))

    print("=== Clean model (with input) ===\n")
    model.diagnose(x)

    print("=== After forward / loss / backward ===\n")
    y = nn.Tensor(np.zeros((8, 2)))
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    report = model.diagnose(x, verbose=False)
    print(f"critical={report.critical} warnings={report.warnings}")
    print(f"codes={[f.code for f in report.findings if f.severity != 'info']}")

    print("\n=== Deliberate dead ReLU ===\n")

    class AlwaysNegative(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, t: nn.Tensor) -> nn.Tensor:
            return t * 0.0 - 1.0

    dead = nn.Sequential(AlwaysNegative(), nn.ReLU())
    dead.diagnose(nn.Tensor(np.ones((4, 4))))


if __name__ == "__main__":
    main()
