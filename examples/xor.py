"""Train a tiny MLP on the XOR function."""

from __future__ import annotations

import numpy as np

import nanonet_ml as nn


def main() -> None:
    nn.manual_seed(42)

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    )
    optimizer = nn.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 2001):
        pred = model(nn.Tensor(X))
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"epoch={epoch:4d}  loss={float(loss.data):.6f}")

    pred = model(nn.Tensor(X))
    print("predictions:", np.round(pred.data.reshape(-1), 3))


if __name__ == "__main__":
    main()
