"""Benchmark an equivalent PyTorch MLP on the same synthetic workload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install with: "
            "pip install 'nanonet-ml[benchmark]' "
            "or pip install torch"
        ) from exc

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyTorch on the NanoNet "
            "synthetic random-label MLP workload."
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
            / "benchmark_pytorch.json"
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    torch.set_default_dtype(
        torch.float64
    )

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

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()

    t0 = time.perf_counter()

    model.train()

    for _ in range(args.epochs):
        perm = torch.randperm(
            args.samples
        )

        for start in range(
            0,
            args.samples,
            args.batch_size,
        ):
            idx = perm[
                start : start + args.batch_size
            ]

            logits = model(
                Xt[idx]
            )

            loss = loss_fn(
                logits,
                yt[idx],
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

    train_s = (
        time.perf_counter() - t0
    )

    model.eval()

    Xtb = torch.from_numpy(
        X_test[:256]
    )

    with torch.no_grad():
        for _ in range(5):
            _ = model(Xtb)

        t1 = time.perf_counter()

        for _ in range(50):
            _ = model(Xtb)

        infer_s = (
            time.perf_counter() - t1
        ) / 50

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    result = {
        "framework": "pytorch",
        "workload": "synthetic-random-labels",
        "dtype": "float64",
        "train_seconds": train_s,
        "inference_seconds_per_batch": infer_s,
        "inference_batch_size": 256,
        "num_parameters": n_params,
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