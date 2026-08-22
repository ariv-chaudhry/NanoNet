"""Simple single-process DataLoader."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from nanonet_ml.data.dataset import Dataset
from nanonet_ml.utils import get_rng


class DataLoader:
    """Mini-batch iterator over a Dataset.

    Features:
        * batching
        * optional shuffling
        * deterministic seeding via ``nanonet_ml.manual_seed`` or ``seed``
        * ``drop_last`` for incomplete final batches

    Multiprocessing is intentionally omitted for educational clarity.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.seed) if self.seed is not None else get_rng()
            rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield self._collate([self.dataset[int(i)] for i in batch_idx])

    def _collate(self, samples: list[Any]) -> Any:
        """Stack samples into batched NumPy arrays."""
        if not samples:
            return samples

        first = samples[0]
        if isinstance(first, tuple):
            cols = list(zip(*samples))
            return tuple(np.stack([np.asarray(x) for x in col]) for col in cols)
        return np.stack([np.asarray(x) for x in samples])