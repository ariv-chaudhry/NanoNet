"""Fit a nonlinear synthetic regression problem and plot the result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import nanonet_ml as nn


def main() -> None:
    nn.manual_seed(0)
    rng = np.random.default_rng(0)

    X = rng.uniform(-2.0, 2.0, size=(200, 1))
    noise = rng.normal(0.0, 0.2, size=(200, 1))
    y = 3 * X**2 + 2 * X + noise

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = nn.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    history = model.fit(
        X,
        y,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=100,
        batch_size=32,
        verbose=True,
    )

    xs = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    model.eval()
    ys = model(nn.Tensor(xs)).data

    results = Path("results")
    results.mkdir(exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X, y, s=12, alpha=0.5, label="training data")
        ax.plot(xs, ys, color="crimson", linewidth=2, label="NanoNet prediction")
        ax.plot(xs, 3 * xs**2 + 2 * xs, "--", color="gray", label="true (no noise)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Nonlinear regression with NanoNet")
        ax.legend()
        fig.tight_layout()
        out = results / "regression.png"
        fig.savefig(out, dpi=150)
        print(f"Saved figure to {out}")
        history.plot(save_path=results / "regression_history.png")
    except ImportError:
        print("Matplotlib not installed; skipping plots. Install with: pip install matplotlib")

    print(f"Final train loss: {history.loss[-1]:.4f}")


if __name__ == "__main__":
    main()
