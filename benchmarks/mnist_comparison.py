"""Matched MNIST learning comparison: NanoNet vs PyTorch on CPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import (  # noqa: E402
    RESULTS_DIR,
    assign_numpy_params_to_nanonet,
    copy_nanonet_weights_to_pytorch,
    environment_metadata,
    nanonet_mlp,
    pytorch_mlp,
    require_torch,
    save_json,
    set_global_seeds,
)

ARCH = (784, 128, 64, 10)


def _init_params(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic NanoNet-layout Kaiming-style weights for ARCH."""
    params: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(ARCH) - 1):
        fan_in, fan_out = ARCH[i], ARCH[i + 1]
        # Match NanoNet's default Kaiming uniform scale approximately.
        limit = np.sqrt(6.0 / fan_in)
        w = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
        b = np.zeros(fan_out, dtype=np.float64)
        params.append((w, b))
    return params


def _batch_indices(n: int, batch_size: int, seed: int, epochs: int) -> list[np.ndarray]:
    """Precompute identical shuffle permutations for both frameworks."""
    rng = np.random.default_rng(seed + 17)
    batches: list[np.ndarray] = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            batches.append(perm[start : start + batch_size])
    return batches


def _train_nanonet_epochs(
    X: np.ndarray,
    y: np.ndarray,
    params: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[object, list[float], float]:
    from nanonet_ml import Tensor
    from nanonet_ml.losses import CrossEntropyLoss
    from nanonet_ml.optimizers import SGD

    model = nanonet_mlp(ARCH)
    assign_numpy_params_to_nanonet(model, params)
    opt = SGD(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    epoch_losses: list[float] = []
    n = X.shape[0]
    batch_lists = _batch_indices(n, batch_size, seed, epochs)

    t0 = time.perf_counter()
    model.train()
    cursor = 0
    for _epoch in range(epochs):
        total = 0.0
        count = 0
        n_batches = (n + batch_size - 1) // batch_size
        for _ in range(n_batches):
            idx = batch_lists[cursor]
            cursor += 1
            xb, yb = X[idx], y[idx]
            logits = model(Tensor(xb))
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.data) * len(idx)
            count += len(idx)
        epoch_losses.append(total / max(count, 1))
    train_s = time.perf_counter() - t0
    return model, epoch_losses, train_s


def _train_pytorch_epochs(
    X: np.ndarray,
    y: np.ndarray,
    params: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[object, list[float], float]:
    torch = require_torch()
    import torch.nn as nn

    torch.set_default_dtype(torch.float64)
    model = pytorch_mlp(ARCH, dtype=torch.float64)
    src = nanonet_mlp(ARCH)
    assign_numpy_params_to_nanonet(src, params)
    copy_nanonet_weights_to_pytorch(src, model)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = X.shape[0]
    batch_lists = _batch_indices(n, batch_size, seed, epochs)
    epoch_losses: list[float] = []

    t0 = time.perf_counter()
    model.train()
    cursor = 0
    for _epoch in range(epochs):
        total = 0.0
        count = 0
        n_batches = (n + batch_size - 1) // batch_size
        for _ in range(n_batches):
            idx = batch_lists[cursor]
            cursor += 1
            idx_t = torch.from_numpy(idx)
            logits = model(Xt[idx_t])
            loss = loss_fn(logits, yt[idx_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * len(idx)
            count += len(idx)
        epoch_losses.append(total / max(count, 1))
    train_s = time.perf_counter() - t0
    return model, epoch_losses, train_s


def _eval_nanonet(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from nanonet_ml import Tensor, no_grad
    from nanonet_ml.metrics import accuracy

    t0 = time.perf_counter()
    model.eval()
    with no_grad():
        logits = model(Tensor(X))
        acc = float(accuracy(logits, y))
    return acc, time.perf_counter() - t0


def _eval_pytorch(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    torch = require_torch()
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        preds = logits.argmax(dim=1).cpu().numpy()
        acc = float(np.mean(preds == y))
    return acc, time.perf_counter() - t0


def run_mnist_comparison(
    *,
    train_samples: int = 5000,
    test_samples: int = 1000,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.1,
    seed: int = 0,
    out: Path | None = None,
    plot: Path | None = None,
) -> dict:
    """Train matching NanoNet and PyTorch MLPs on the same MNIST subset."""
    require_torch()
    set_global_seeds(seed)

    from nanonet_ml.data import load_mnist

    print("Loading MNIST...")
    try:
        X_train, y_train, X_test, y_test = load_mnist()
    except Exception as exc:
        raise SystemExit(
            f"Failed to load MNIST: {exc}\n"
            "Run: python scripts/download_mnist.py"
        ) from exc

    X_train = X_train[:train_samples].astype(np.float64)
    y_train = y_train[:train_samples].astype(np.int64)
    X_test = X_test[:test_samples].astype(np.float64)
    y_test = y_test[:test_samples].astype(np.int64)

    rng = np.random.default_rng(seed)
    params = _init_params(rng)

    print("Training NanoNet...")
    nn_model, nn_losses, nn_train_s = _train_nanonet_epochs(
        X_train,
        y_train,
        params,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    nn_acc, nn_eval_s = _eval_nanonet(nn_model, X_test, y_test)

    print("Training PyTorch...")
    pt_model, pt_losses, pt_train_s = _train_pytorch_epochs(
        X_train,
        y_train,
        params,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    pt_acc, pt_eval_s = _eval_pytorch(pt_model, X_test, y_test)

    payload = {
        "experiment": "mnist_comparison",
        "environment": environment_metadata(
            dtype="float64",
            device="CPU",
            seed=seed,
            extra={
                "architecture": "784 → 128 → 64 → 10",
                "optimizer": f"SGD(lr={lr})",
                "loss": "CrossEntropyLoss(mean)",
            },
        ),
        "configuration": {
            "train_samples": train_samples,
            "test_samples": test_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "identical_init": True,
            "identical_batch_order": True,
        },
        "results": {
            "nanonet": {
                "test_accuracy": nn_acc,
                "train_seconds": nn_train_s,
                "eval_seconds": nn_eval_s,
                "epoch_losses": nn_losses,
            },
            "pytorch": {
                "test_accuracy": pt_acc,
                "train_seconds": pt_train_s,
                "eval_seconds": pt_eval_s,
                "epoch_losses": pt_losses,
            },
            "accuracy_difference": abs(nn_acc - pt_acc),
            "final_loss_difference": abs(nn_losses[-1] - pt_losses[-1]) if nn_losses and pt_losses else None,
        },
    }

    out_path = Path(out) if out is not None else RESULTS_DIR / "mnist_comparison.json"
    save_json(out_path, payload)
    payload["_out_path"] = str(out_path)

    plot_path = Path(plot) if plot is not None else RESULTS_DIR / "mnist_comparison.png"
    try:
        _plot_mnist(payload, plot_path, epochs=epochs)
        payload["_plot_path"] = str(plot_path)
    except ImportError:
        print("Matplotlib not installed; skipping MNIST plot.")

    print_report(payload)
    print(f"\nSaved: {out_path}")
    return payload


def _plot_mnist(payload: dict, path: Path, *, epochs: int) -> None:
    import matplotlib.pyplot as plt

    nn = payload["results"]["nanonet"]
    pt = payload["results"]["pytorch"]

    if epochs > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = list(range(1, epochs + 1))
        ax.plot(xs, nn["epoch_losses"], marker="o", label="NanoNet")
        ax.plot(xs, pt["epoch_losses"], marker="s", label="PyTorch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("MNIST training loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].bar(["NanoNet", "PyTorch"], [nn["test_accuracy"] * 100, pt["test_accuracy"] * 100])
        axes[0].set_ylabel("Test accuracy (%)")
        axes[0].set_title("MNIST test accuracy")
        axes[1].bar(["NanoNet", "PyTorch"], [nn["train_seconds"], pt["train_seconds"]])
        axes[1].set_ylabel("Training time (s)")
        axes[1].set_title("MNIST training runtime")
        fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_report(payload: dict) -> None:
    cfg = payload["configuration"]
    nn = payload["results"]["nanonet"]
    pt = payload["results"]["pytorch"]
    print("NanoNet vs PyTorch - MNIST")
    print("=" * 27)
    print()
    print("Configuration")
    print("-------------")
    print(f"Train samples: {cfg['train_samples']}")
    print(f"Test samples: {cfg['test_samples']}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print("Architecture: 784 -> 128 -> 64 -> 10")
    print(f"Optimizer: SGD (lr={cfg['lr']})")
    print()
    print("Results")
    print("-------")
    print(f"{'Framework':<10} {'Accuracy':>10} {'Train Time':>12} {'Eval Time':>12}")
    print(
        f"{'NanoNet':<10} {nn['test_accuracy'] * 100:>9.2f}% "
        f"{nn['train_seconds']:>10.2f} s {nn['eval_seconds']:>10.2f} s"
    )
    print(
        f"{'PyTorch':<10} {pt['test_accuracy'] * 100:>9.2f}% "
        f"{pt['train_seconds']:>10.2f} s {pt['eval_seconds']:>10.2f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched NanoNet vs PyTorch MNIST comparison.")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "mnist_comparison.json")
    parser.add_argument("--plot", type=Path, default=RESULTS_DIR / "mnist_comparison.png")
    args = parser.parse_args()

    run_mnist_comparison(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out=args.out,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
