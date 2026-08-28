"""Dataset abstraction for indexable sample collections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(ABC, Generic[T_co]):
    """Indexable collection of samples.

    Subclasses define how samples are stored or generated. A sample may be a
    single value (for example features) or a structured value such as
    ``(features, target)``.

    This layer does not convert samples into tensors; batching and tensor
    conversion belong to a separate data-loading step.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Return the sample at ``index``.

        Negative indices and bounds checking are left to subclasses.
        """


class TensorDataset(Dataset[Any]):
    """Dataset wrapping aligned NumPy arrays or sequences.

    Example::

        ds = TensorDataset(X, y)
        x0, y0 = ds[0]
    """

    def __init__(self, *arrays: Any) -> None:
        if not arrays:
            raise ValueError("TensorDataset requires at least one array.")
        self.arrays = arrays
        length = len(arrays[0])
        for i, arr in enumerate(arrays):
            if len(arr) != length:
                raise ValueError(
                    f"TensorDataset arrays must share length; "
                    f"array 0 has {length}, array {i} has {len(arr)}."
                )

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index: int) -> tuple[Any, ...] | Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
        items = tuple(arr[index] for arr in self.arrays)
        return items[0] if len(items) == 1 else items
