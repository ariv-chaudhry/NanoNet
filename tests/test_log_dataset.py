"""Tests for LogDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanonet_ml.data import DataLoader, LogDataset


def _write_log(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_feature_only_parsing(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200 12\nERROR 500 83\n")

    def parse(line: str):
        level, status, latency = line.split()
        levels = {"INFO": 0.0, "ERROR": 2.0}
        return [levels[level], float(status), float(latency)]

    dataset = LogDataset(path, parser=parse)
    assert len(dataset) == 2
    assert dataset[0] == [0.0, 200.0, 12.0]
    assert dataset[1] == [2.0, 500.0, 83.0]


def test_supervised_parsing(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200\nERROR 500\n")

    def parse(line: str):
        level, status = line.split()
        levels = {"INFO": 0.0, "ERROR": 2.0}
        features = [levels[level], float(status)]
        label = 0 if level == "INFO" else 1
        return features, label

    dataset = LogDataset(path, parser=parse)
    assert dataset[0] == ([0.0, 200.0], 0)
    assert dataset[1] == ([2.0, 500.0], 1)


def test_parser_receives_line_without_newline(tmp_path: Path):
    path = tmp_path / "mixed.log"
    # Explicit LF and CRLF terminators in the written bytes.
    path.write_bytes(b"INFO test\nWARNING other\r\n")

    seen: list[str] = []

    def parse(line: str):
        seen.append(line)
        return line

    dataset = LogDataset(path, parser=parse)
    assert dataset[0] == "INFO test"
    assert dataset[1] == "WARNING other"
    assert seen == ["INFO test", "WARNING other"]
    assert not any(s.endswith("\n") or s.endswith("\r") for s in seen)


def test_lazy_parsing(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200\nERROR 500\n")
    calls = {"n": 0}

    def parse(line: str):
        calls["n"] += 1
        return line

    dataset = LogDataset(path, parser=parse)
    assert calls["n"] == 0
    _ = dataset[0]
    assert calls["n"] == 1
    _ = dataset[1]
    assert calls["n"] == 2


def test_negative_indexing(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "a\nb\nc\n")
    dataset = LogDataset(path, parser=lambda line: line)
    assert dataset[-1] == "c"
    assert dataset[-2] == "b"


def test_invalid_indices(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "only\n")
    dataset = LogDataset(path, parser=lambda line: line)
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[1]
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[-2]


def test_empty_file(tmp_path: Path):
    path = _write_log(tmp_path / "empty.log", "")
    dataset = LogDataset(path, parser=lambda line: line)
    assert len(dataset) == 0


def test_blank_line_not_skipped(tmp_path: Path):
    path = _write_log(tmp_path / "blank.log", "INFO\n\nERROR\n")
    dataset = LogDataset(path, parser=lambda line: repr(line))
    assert len(dataset) == 3
    assert dataset[1] == "''"


def test_path_object(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "x\n")
    dataset = LogDataset(Path(path), parser=lambda line: line)
    assert dataset[0] == "x"


def test_missing_file(tmp_path: Path):
    missing = tmp_path / "does-not-exist.log"
    with pytest.raises(FileNotFoundError):
        LogDataset(missing, parser=lambda line: line)


def test_non_callable_parser(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "x\n")
    with pytest.raises(TypeError, match="parser must be callable"):
        LogDataset(path, parser="abc")  # type: ignore[arg-type]


def test_public_import():
    from nanonet_ml.data import LogDataset as Exported

    assert Exported is LogDataset


def test_dataloader_feature_batches(tmp_path: Path):
    path = _write_log(
        tmp_path / "server.log",
        "INFO 200 12\nERROR 500 83\nWARNING 400 40\nINFO 201 10\n",
    )

    def parse(line: str):
        level, status, latency = line.split()
        levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
        return [levels[level], float(status), float(latency)]

    loader = DataLoader(LogDataset(path, parser=parse), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert isinstance(batches[0], np.ndarray)
    assert batches[0].shape == (2, 3)
    assert np.allclose(batches[0], [[0.0, 200.0, 12.0], [2.0, 500.0, 83.0]])


def test_dataloader_supervised_batches(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200\nERROR 500\nWARNING 400\n")

    def parse(line: str):
        level, status = line.split()
        levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
        return [levels[level], float(status)], 1 if level == "ERROR" else 0

    loader = DataLoader(LogDataset(path, parser=parse), batch_size=2, shuffle=False)
    features, labels = next(iter(loader))
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert features.shape == (2, 2)
    assert labels.shape == (2,)
    assert np.allclose(features, [[0.0, 200.0], [2.0, 500.0]])
    assert np.array_equal(labels, [0, 1])
