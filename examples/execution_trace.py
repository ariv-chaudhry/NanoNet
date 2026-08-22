"""Demonstrate NanoNet execution tracing."""

from __future__ import annotations

import numpy as np

import nanonet_ml as nn


def main() -> None:
    nn.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = nn.Tensor(np.random.randn(2, 4))

    print("=== Execution trace ===\n")
    trace = model.trace(x)

    print("=== Programmatic access ===")
    for step in trace.steps:
        in_ids = [t.trace_id for t in step.inputs]
        out_ids = [t.trace_id for t in step.outputs]
        print(
            f"{step.index}: {step.module_name} ({step.module_type}) "
            f"{in_ids} -> {out_ids}  params={step.parameter_count}"
        )

    print("\n=== Autograd still works from traced output ===")
    loss = nn.MSELoss()(trace.output, nn.Tensor(np.zeros((2, 2))))
    loss.backward()
    print(f"loss={float(loss.data):.4f}")
    print(f"grad norm (first weight)={float(np.linalg.norm(model[0].weight.grad)):.4f}")


if __name__ == "__main__":
    main()
