"""End-to-end NanoNet observability workflow."""

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
    x = nn.Tensor(np.random.randn(4, 4))
    target = nn.Tensor(np.zeros((4, 2)))

    print("1) Model inspection\n")
    model.inspect(x)

    print("2) Execution tracing\n")
    trace = model.trace(x, verbose=False)
    print(f"   traced {len(trace.steps)} leaf steps; output shape={trace.output.shape}")

    print("\n3) Forward, loss, and computation graph\n")
    prediction = model(x)
    loss = nn.MSELoss()(prediction, target)
    loss.graph()

    print("4) Backward + diagnostics\n")
    loss.backward()
    report = model.diagnose(x, verbose=False)
    print(f"   critical={report.critical} warnings={report.warnings} ok={report.ok}")
    print(f"   finding codes: {[f.code for f in report.findings if f.severity != 'info']}")


if __name__ == "__main__":
    main()
