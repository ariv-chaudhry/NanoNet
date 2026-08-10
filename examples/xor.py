"""Train a tiny MLP on the XOR function."""

from __future__ import annotations

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, Tanh
from nanonet.losses import MSELoss
from nanonet.optimizers import Adam


def main() -> None:
    manual_seed(42)

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = Sequential(
        Dense(2, 8),
        Tanh(),
        Dense(8, 1),
    )
    optimizer = Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()

    for epoch in range(1, 2001):
        pred = model(Tensor(X))
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss={float(loss.data):.6f}")

    model.eval()
    preds = model(Tensor(X)).data
    bits = (preds >= 0.5).astype(int).reshape(-1)

    print("\nInput     Prediction")
    for (a, b), p, bit in zip(X, preds.reshape(-1), bits):
        print(f"{int(a)} {int(b)}       {bit}  (raw={p:.4f})")

    assert np.array_equal(bits, y.reshape(-1).astype(int)), "XOR was not learned."
    print("\nXOR learned successfully.")


if __name__ == "__main__":
    main()
