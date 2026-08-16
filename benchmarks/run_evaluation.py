"""Unified evaluation runner for NanoNet vs PyTorch experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NanoNet empirical evaluation suite against PyTorch."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Numerical parity + small multi-trial runtime comparison.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Parity + scaling + MNIST (slower).",
    )
    args = parser.parse_args()

    if not args.quick and not args.full:
        parser.print_help()
        print("\nChoose --quick or --full.")
        raise SystemExit(2)

    from benchmarks.numerical_parity import print_report, run_parity

    print("=== Numerical parity ===\n")
    parity = run_parity()
    print_report(parity)
    if not parity["results"]["overall_pass"]:
        raise SystemExit(1)

    if args.quick:
        from benchmarks.scaling_benchmark import run_scaling

        print("\n=== Quick runtime (n=1000, 3 runs) ===\n")
        run_scaling(sizes=[1000], epochs=1, warmup=1, runs=3)
        return

    # --full
    from benchmarks.mnist_comparison import run_mnist_comparison
    from benchmarks.scaling_benchmark import run_scaling

    print("\n=== Runtime scaling ===\n")
    run_scaling(sizes=[1000, 5000, 10000], epochs=1, warmup=1, runs=5)

    print("\n=== MNIST comparison ===\n")
    run_mnist_comparison(train_samples=5000, test_samples=1000, epochs=1)


if __name__ == "__main__":
    main()
