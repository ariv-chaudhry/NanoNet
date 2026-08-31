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

    Args:
        path: Path to a line-oriented text file (any extension).
        parser: Callable that receives one logical line and returns a sample.
            Output may be feature-only or ``(features, target)``; structure is
            unrestricted except by downstream DataLoader compatibility.
        encoding: Text encoding used to decode the file. Defaults to ``"utf-8"``.
        skip_blank_lines: If ``True``, omit blank and whitespace-only physical
            lines from the dataset. Defaults to ``False`` (Stage 1 behavior).

    Notes:
        * One kept logical line corresponds to one dataset record.
        * Newline terminators (``\\n``, ``\\r\\n``) are removed; other whitespace
          on nonblank lines is preserved.
        * The source file is read and indexed at construction time. The
          user-provided ``parser`` is applied lazily in ``__getitem__``.
        * Intended for files that reasonably fit in memory.
        * Blank lines are preserved by default.
        * Parser output is unrestricted except by downstream DataLoader compatibility.
        * Parser failures are reported with source location context.

    Example::

        def parse(line: str):
            level, status = line.split()
            levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
            return [levels[level], float(status)]

        dataset = LogDataset(
            "server.log",
            parser=parse,
            encoding="utf-8",
            skip_blank_lines=True,
        )
    """

    def __init__(
        self,
        path: str | Path,
        parser: Callable[[str], Any],
        *,
        encoding: str = "utf-8",
        skip_blank_lines: bool = False,
    ) -> None:
        if not callable(parser):
            raise TypeError(f"parser must be callable, got {type(parser).__name__}.")
        path = Path(path)
        # FileNotFoundError / UnicodeDecodeError / LookupError propagate normally.
        text = path.read_text(encoding=encoding)
        # splitlines() drops \\n / \\r\\n without stripping other whitespace.
        records: list[tuple[int, str]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if skip_blank_lines and line.strip() == "":
                continue
            records.append((line_no, line))
        self._records = records
        self._path = path
        self.parser = parser

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
        line_no, line = self._records[index]
        try:
            return self.parser(line)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse log record at line {line_no} in '{self._path}' "
                f"(dataset index {index})."
            ) from exc
