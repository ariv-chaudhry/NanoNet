"""Dataset-size (and optional batch-size) scaling: NanoNet vs PyTorch on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import (  # noqa: E402
    RESULTS_DIR,
    TimingStats,
    environment_metadata,
    measure_repeated,
    nanonet_mlp,
    pytorch_mlp,
    require_torch,
    save_json,
    set_global_seeds,
    slowdown,
)

ARCH = (784, 128, 64, 10)


def _make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 784), dtype=np.float64)
    y = rng.integers(0, 10, size=(n,), dtype=np.int64)
    X_infer = rng.random((256, 784), dtype=np.float64)
    return X, y, X_infer


def _time_nanonet_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    from nanonet import manual_seed
    from nanonet.losses import CrossEntropyLoss
    from nanonet.optimizers import Adam

    manual_seed(seed)
    model = nanonet_mlp(ARCH)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    model.fit(
        X,
        y,
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
        compute_accuracy=False,
    )


def _time_nanonet_infer(X_infer: np.ndarray, *, seed: int) -> None:
    from nanonet import Tensor, manual_seed, no_grad

    manual_seed(seed)
    model = nanonet_mlp(ARCH)
    model.eval()
    batch = Tensor(X_infer)
    with no_grad():
        _ = model(batch)


def _time_pytorch_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    torch = require_torch()
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    model = pytorch_mlp(ARCH, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    model.train()
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = loss_fn(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def _time_pytorch_infer(X_infer: np.ndarray, *, seed: int) -> None:
    torch = require_torch()
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    model = pytorch_mlp(ARCH, dtype=torch.float64)
    model.eval()
    xb = torch.from_numpy(X_infer)
    with torch.no_grad():
        _ = model(xb)


def _stats_dict(stats: TimingStats) -> dict:
    return stats.to_dict()


def run_scaling(
    *,
    sizes: list[int],
    batch_sizes: list[int] | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    warmup: int = 1,
    runs: int = 5,
    seed: int = 0,
    out: Path | None = None,
    plot: Path | None = None,
) -> dict:
    """Measure training/inference runtime across dataset sizes (CPU, float64)."""
    require_torch()
    set_global_seeds(seed)

    size_results = []
    for n in sizes:
        print(f"\nDataset size n={n} ...")
        X, y, X_infer = _make_data(n, seed)

        nn_train = measure_repeated(
            lambda X=X, y=y: _time_nanonet_train(
                X, y, epochs=epochs, batch_size=batch_size, seed=seed
            ),
            warmup=warmup,
            runs=runs,
        )
        pt_train = measure_repeated(
            lambda X=X, y=y: _time_pytorch_train(
                X, y, epochs=epochs, batch_size=batch_size, seed=seed
            ),
            warmup=warmup,
            runs=runs,
        )
        nn_infer = measure_repeated(
            lambda Xi=X_infer: _time_nanonet_infer(Xi, seed=seed),
            warmup=warmup,
            runs=runs,
        )
        pt_infer = measure_repeated(
            lambda Xi=X_infer: _time_pytorch_infer(Xi, seed=seed),
            warmup=warmup,
            runs=runs,
        )

        row = {
            "samples": n,
            "nanonet_train": _stats_dict(nn_train),
            "pytorch_train": _stats_dict(pt_train),
            "nanonet_infer": _stats_dict(nn_infer),
            "pytorch_infer": _stats_dict(pt_infer),
            "train_slowdown": slowdown(nn_train.mean, pt_train.mean),
            "infer_slowdown": slowdown(nn_infer.mean, pt_infer.mean),
        }
        size_results.append(row)
        print(
            f"  train  NanoNet {nn_train.mean:.4f}s +/- {nn_train.std:.4f} | "
            f"PyTorch {pt_train.mean:.4f}s +/- {pt_train.std:.4f} | "
            f"slowdown {row['train_slowdown']:.1f}x"
        )
        print(
            f"  infer  NanoNet {nn_infer.mean:.5f}s +/- {nn_infer.std:.5f} | "
            f"PyTorch {pt_infer.mean:.5f}s +/- {pt_infer.std:.5f} | "
            f"slowdown {row['infer_slowdown']:.1f}x"
        )

    batch_results = []
    if batch_sizes:
        # Hold dataset size fixed at the median / first listed size.
        fixed_n = sizes[min(1, len(sizes) - 1)]
        X, y, X_infer = _make_data(fixed_n, seed)
        for bs in batch_sizes:
            print(f"\nBatch size bs={bs} (n={fixed_n}) ...")
            nn_train = measure_repeated(
                lambda bs=bs: _time_nanonet_train(
                    X, y, epochs=epochs, batch_size=bs, seed=seed
                ),
                warmup=warmup,
                runs=runs,
            )
            pt_train = measure_repeated(
                lambda bs=bs: _time_pytorch_train(
                    X, y, epochs=epochs, batch_size=bs, seed=seed
                ),
                warmup=warmup,
                runs=runs,
            )
            batch_results.append(
                {
                    "batch_size": bs,
                    "samples": fixed_n,
                    "nanonet_train": _stats_dict(nn_train),
                    "pytorch_train": _stats_dict(pt_train),
                    "train_slowdown": slowdown(nn_train.mean, pt_train.mean),
                }
            )

    payload = {
        "experiment": "runtime_scaling",
        "environment": environment_metadata(
            dtype="float64",
            device="CPU",
            seed=seed,
            extra={
                "architecture": "784 → 128 → 64 → 10",
                "optimizer": "Adam(lr=1e-3)",
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_runs": warmup,
                "measured_runs": runs,
                "inference_batch_size": 256,
            },
        ),
        "configuration": {
            "sizes": sizes,
            "batch_sizes": batch_sizes,
            "epochs": epochs,
            "batch_size": batch_size,
            "warmup": warmup,
            "runs": runs,
            "seed": seed,
        },
        "results": {
            "by_dataset_size": size_results,
            "by_batch_size": batch_results,
        },
    }

    out_path = Path(out) if out is not None else RESULTS_DIR / "runtime_scaling.json"
    save_json(out_path, payload)
    payload["_out_path"] = str(out_path)

    plot_path = Path(plot) if plot is not None else RESULTS_DIR / "runtime_scaling.png"
    try:
        _plot_scaling(size_results, plot_path)
        payload["_plot_path"] = str(plot_path)
        print(f"\nSaved plot: {plot_path}")
    except ImportError:
        print("\nMatplotlib not installed; skipping plot. pip install matplotlib")

    print(f"Saved JSON: {out_path}")
    return payload


def _plot_scaling(size_results: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    xs = [r["samples"] for r in size_results]
    nn_means = [r["nanonet_train"]["mean"] for r in size_results]
    nn_stds = [r["nanonet_train"]["std"] for r in size_results]
    pt_means = [r["pytorch_train"]["mean"] for r in size_results]
    pt_stds = [r["pytorch_train"]["std"] for r in size_results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(xs, nn_means, yerr=nn_stds, marker="o", label="NanoNet", capsize=3)
    axes[0].errorbar(xs, pt_means, yerr=pt_stds, marker="s", label="PyTorch", capsize=3)
    axes[0].set_xlabel("Dataset size (samples)")
    axes[0].set_ylabel("Training time (s)")
    axes[0].set_title("Training runtime vs dataset size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    nn_i = [r["nanonet_infer"]["mean"] for r in size_results]
    nn_is = [r["nanonet_infer"]["std"] for r in size_results]
    pt_i = [r["pytorch_infer"]["mean"] for r in size_results]
    pt_is = [r["pytorch_infer"]["std"] for r in size_results]
    # Inference uses a fixed 256-batch; plot vs the training set size index for context
    # but label clearly that inference batch is constant.
    axes[1].errorbar(xs, nn_i, yerr=nn_is, marker="o", label="NanoNet", capsize=3)
    axes[1].errorbar(xs, pt_i, yerr=pt_is, marker="s", label="PyTorch", capsize=3)
    axes[1].set_xlabel("Associated training set size")
    axes[1].set_ylabel("Inference time (s / 256-batch)")
    axes[1].set_title("Inference runtime (fixed batch=256)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoNet vs PyTorch runtime scaling benchmark.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional batch-size sweep (holds dataset size fixed).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "runtime_scaling.json")
    parser.add_argument("--plot", type=Path, default=RESULTS_DIR / "runtime_scaling.png")
    args = parser.parse_args()

    run_scaling(
        sizes=list(args.sizes),
        batch_sizes=list(args.batch_sizes) if args.batch_sizes else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
        out=args.out,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
