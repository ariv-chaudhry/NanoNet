"""Dataset abstraction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Dataset(Protocol):
    """Minimal dataset protocol inspired by common ML frameworks."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class TensorDataset:
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