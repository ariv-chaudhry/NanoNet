"""Compare NanoNet and PyTorch benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare NanoNet vs PyTorch benchmarks.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-run", action="store_true", help="Only compare existing JSON files.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    nn_out = root.parent / "results" / "benchmark_nanonet.json"
    pt_out = root.parent / "results" / "benchmark_pytorch.json"

    if not args.skip_run:
        subprocess.check_call(
            [
                sys.executable,
                str(root / "benchmark_nanonet.py"),
                "--samples",
                str(args.samples),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--out",
                str(nn_out),
            ]
        )
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    str(root / "benchmark_pytorch.py"),
                    "--samples",
                    str(args.samples),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--out",
                    str(pt_out),
                ]
            )
        except subprocess.CalledProcessError:
            print("PyTorch benchmark failed (is torch installed?). NanoNet results were saved.")
            print(nn_out.read_text(encoding="utf-8"))
            return

    nn = json.loads(nn_out.read_text(encoding="utf-8"))
    pt = json.loads(pt_out.read_text(encoding="utf-8")) if pt_out.exists() else None

    print("\n=== Benchmark comparison ===")
    print("PyTorch is expected to be faster — NanoNet prioritizes educational clarity.\n")
    print(f"{'Framework':<12} {'Params':>10} {'Train(s)':>10} {'Infer(s)':>10} {'Acc':>8}")
    print("-" * 54)
    print(
        f"{'NanoNet':<12} {nn['num_parameters']:>10} {nn['train_seconds']:>10.3f} "
        f"{nn['inference_seconds_per_batch']:>10.5f} {nn['test_accuracy']:>8.3f}"
    )
    if pt:
        print(
            f"{'PyTorch':<12} {pt['num_parameters']:>10} {pt['train_seconds']:>10.3f} "
            f"{pt['inference_seconds_per_batch']:>10.5f} {pt['test_accuracy']:>8.3f}"
        )


if __name__ == "__main__":
    main()
