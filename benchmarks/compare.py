"""Compare NanoNet and PyTorch via multi-trial CPU runtime methodology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Local ``benchmarks`` helpers only. Install NanoNet with ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import RESULTS_DIR, set_global_seeds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-trial NanoNet vs PyTorch CPU runtime comparison."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "runtime_comparison.json")
    args = parser.parse_args()

    from benchmarks.scaling_benchmark import run_scaling

    set_global_seeds(args.seed)
    payload = run_scaling(
        sizes=[args.samples],
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
        out=args.out,
        plot=RESULTS_DIR / "runtime_comparison.png",
    )

    row = payload["results"]["by_dataset_size"][0]
    print("\n=== Runtime comparison (CPU, float64) ===")
    print("device: CPU")
    print("dtype: float64")
    print(f"samples: {args.samples}  epochs: {args.epochs}  batch_size: {args.batch_size}")
    print(f"warmup: {args.warmup}  measured_runs: {args.runs}\n")
    print(f"{'Framework':<10} {'Train mean':>12} {'Train std':>10} {'Infer mean':>12}")
    print("-" * 50)
    print(
        f"{'NanoNet':<10} {row['nanonet_train']['mean']:>11.4f}s "
        f"{row['nanonet_train']['std']:>9.4f} "
        f"{row['nanonet_infer']['mean']:>11.5f}s"
    )
    print(
        f"{'PyTorch':<10} {row['pytorch_train']['mean']:>11.4f}s "
        f"{row['pytorch_train']['std']:>9.4f} "
        f"{row['pytorch_infer']['mean']:>11.5f}s"
    )
    print(
        f"\nTrain slowdown: {row['train_slowdown']:.1f}x  |  "
        f"Infer slowdown: {row['infer_slowdown']:.1f}x"
    )
    print("\nPyTorch is expected to be faster; NanoNet prioritizes educational clarity.")


if __name__ == "__main__":
    main()
