"""MNIST download and loading utilities.

Downloads the original IDX files from a public mirror, caches them under
``data/mnist/``, and returns flattened, normalized arrays suitable for an MLP.
"""

from __future__ import annotations

import gzip
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# Yann LeCun's original MNIST hosting (mirrored).
_BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _default_root() -> Path:
    return Path.cwd() / "data" / "mnist"


def _download_file(filename: str, dest: Path, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    last_error: Exception | None = None
    for base in _BASE_URLS:
        url = base + filename
        try:
            print(f"Downloading {filename} from {url} ...")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            dest.write_bytes(data)
            print(f"Saved to {dest}")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to download MNIST file '{filename}'. "
        f"Check your network connection or place the file manually at:\n  {dest}\n"
        f"Last error: {last_error}"
    )


def download_mnist(root: str | Path | None = None) -> Path:
    """Download MNIST IDX files into ``root`` if missing.

    Returns:
        Path to the cache directory.
    """
    root_path = Path(root) if root is not None else _default_root()
    root_path.mkdir(parents=True, exist_ok=True)
    for filename in _FILES.values():
        _download_file(filename, root_path / filename)
    return root_path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic number: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist(
    root: str | Path | None = None,
    *,
    download: bool = True,
    flatten: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the MNIST dataset.

    Args:
        root: Cache directory (default: ``./data/mnist``).
        download: Download files if missing.
        flatten: If True, reshape images to ``(N, 784)``.
        normalize: If True, scale pixels to ``[0, 1]``.

    Returns:
        ``(X_train, y_train, X_test, y_test)``
    """
    root_path = Path(root) if root is not None else _default_root()

    if download:
        download_mnist(root_path)
    else:
        for filename in _FILES.values():
            path = root_path / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"MNIST file not found: {path}. Re-run with download=True "
                    f"or execute scripts/download_mnist.py."
                )

    X_train = _read_idx_images(root_path / _FILES["train_images"])
    y_train = _read_idx_labels(root_path / _FILES["train_labels"])
    X_test = _read_idx_images(root_path / _FILES["test_images"])
    y_test = _read_idx_labels(root_path / _FILES["test_labels"])

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)

    if normalize:
        X_train = X_train.astype(np.float64) / 255.0
        X_test = X_test.astype(np.float64) / 255.0
    else:
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

    return X_train, y_train, X_test, y_test