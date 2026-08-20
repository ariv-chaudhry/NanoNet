"""Demonstrate NanoNet computation-graph inspection."""

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
    target = Tensor(np.zeros((2, 2)))

    pred = model(x)
    loss = MSELoss()(pred, target)

    print("=== Computation graph (before backward) ===\n")
    graph = loss.graph()

    print("=== Programmatic access ===")
    print(f"root={graph.root_id} depth={graph.depth} ops={len(graph.operations)}")
    print(f"parameters={[n.id for n in graph.tensors if n.is_parameter]}")
    print(f"op names={[op.name for op in graph.operations]}")

    print("\n=== After backward (grad metadata) ===\n")
    loss.backward()
    after = loss.graph(verbose=False)
    for node in after.tensors:
        if node.is_parameter or node.is_root:
            print(
                f"{node.id}: has_grad={node.has_grad} "
                f"shape={node.shape} parameter={node.is_parameter}"
            )


if __name__ == "__main__":
    main()
