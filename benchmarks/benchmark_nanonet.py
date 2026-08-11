"""Benchmark NanoNet MLP training and inference on a synthetic workload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nanonet import Sequential, Tensor, manual_seed, no_grad
from nanonet.layers import Dense, ReLU
from nanonet.losses import CrossEntropyLoss
from nanonet.optimizers import Adam


def build_model() -> Sequential:
    return Sequential(
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NanoNet on a synthetic "
            "random-label MLP workload."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            Path("results")
            / "benchmark_nanonet.json"
        ),
    )

    args = parser.parse_args()

    manual_seed(args.seed)
    rng = np.random.default_rng(
        args.seed
    )

    X = rng.random(
        (args.samples, 784),
        dtype=np.float64,
    )

    y = rng.integers(
        0,
        10,
        size=(args.samples,),
    )

    X_test = rng.random(
        (1000, 784),
        dtype=np.float64,
    )

    model = build_model()

    opt = Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = CrossEntropyLoss()

    t0 = time.perf_counter()

    model.fit(
        X,
        y,
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=False,
        compute_accuracy=False,
    )

    train_s = (
        time.perf_counter() - t0
    )

    model.eval()

    inference_batch = Tensor(
        X_test[:256]
    )

    with no_grad():
        # Warm up interpreter/BLAS paths before timing.
        for _ in range(5):
            _ = model(
                inference_batch
            )

        t1 = time.perf_counter()

        for _ in range(50):
            _ = model(
                inference_batch
            )

        infer_s = (
            time.perf_counter() - t1
        ) / 50

    result = {
        "framework": "nanonet",
        "workload": "synthetic-random-labels",
        "dtype": "float64",
        "train_seconds": train_s,
        "inference_seconds_per_batch": infer_s,
        "inference_batch_size": 256,
        "num_parameters": model.num_parameters(),
        "samples": args.samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()