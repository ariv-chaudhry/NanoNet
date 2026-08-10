"""Download MNIST into ./data/mnist."""

from __future__ import annotations

import argparse
from pathlib import Path

from nanonet.data import download_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the MNIST dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "mnist",
        help="Destination directory.",
    )
    args = parser.parse_args()
    path = download_mnist(args.root)
    print(f"MNIST ready at {path.resolve()}")


if __name__ == "__main__":
    main()
