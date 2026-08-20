"""Demonstrate NanoNet execution tracing."""

from __future__ import annotations

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss


def main() -> None:
    manual_seed(0)
    model = Sequential(
        Dense(4, 8),
        ReLU(),
        Dense(8, 2),
    )
    x = Tensor(np.random.randn(2, 4))

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
    loss = MSELoss()(trace.output, Tensor(np.zeros((2, 2))))
    loss.backward()
    print(f"loss={float(loss.data):.4f}")
    print(f"grad norm (first weight)={float(np.linalg.norm(model[0].weight.grad)):.4f}")


if __name__ == "__main__":
    main()
