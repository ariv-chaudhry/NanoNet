"""Line-oriented log file dataset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class LogDataset:
    """Dataset that maps each logical line of a text log file to a sample.

    NanoNet does not interpret log formats. You supply a ``parser`` that
    converts one line (without trailing newline characters) into a numerical
    sample compatible with :class:`~nanonet_ml.data.dataloader.DataLoader`.

    Parsing is lazy: lines are loaded at construction time, but ``parser`` runs
    only when a sample is accessed via ``__getitem__``.

    Parser output may be feature-only (e.g. a list of floats) or supervised
    ``(features, target)``. :class:`~nanonet_ml.data.dataloader.DataLoader`
    collates samples into NumPy batches using its existing behavior.

    Example::

        def parse(line: str):
            level, status = line.split()
            levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
            return [levels[level], float(status)]

        dataset = LogDataset("server.log", parser=parse)
    """

    def __init__(
        self,
        path: str | Path,
        parser: Callable[[str], Any],
    ) -> None:
        if not callable(parser):
            raise TypeError(f"parser must be callable, got {type(parser).__name__}.")
        path = Path(path)
        # UTF-8; FileNotFoundError propagates for missing paths.
        text = path.read_text(encoding="utf-8")
        # splitlines() drops \\n / \\r\\n without stripping other whitespace.
        self._lines = text.splitlines()
        self._path = path
        self.parser = parser

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
        return self.parser(self._lines[index])
