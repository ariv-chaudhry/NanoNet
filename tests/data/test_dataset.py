"""Tests for the Dataset abstraction."""

from __future__ import annotations

import pytest

from nanonet_ml.data import Dataset


def test_dataset_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Dataset()


def test_simple_subclass():
    class SimpleDataset(Dataset[int]):
        def __init__(self) -> None:
            self.samples = [1, 2, 3]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> int:
            return self.samples[index]

    dataset = SimpleDataset()
    assert len(dataset) == 3
    assert dataset[0] == 1
    assert dataset[2] == 3


def test_tuple_samples_supported():
    class PairDataset(Dataset[tuple[list[float], int]]):
        def __init__(self) -> None:
            self.samples = [
                ([1.0, 2.0], 0),
                ([3.0, 4.0], 1),
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[list[float], int]:
            return self.samples[index]

    dataset = PairDataset()
    assert len(dataset) == 2
    assert dataset[0] == ([1.0, 2.0], 0)
    assert dataset[1] == ([3.0, 4.0], 1)


def test_package_export():
    assert Dataset is not None


def test_nn_data_submodule_export():
    import nanonet_ml as nn

    assert nn.data.Dataset is Dataset
    assert not hasattr(nn, "Dataset")
