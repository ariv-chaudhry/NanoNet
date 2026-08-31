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


# --- Stage 2: robustness -------------------------------------------------


def test_default_utf8_compatible_without_encoding_kwarg(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200\n")
    dataset = LogDataset(path, parser=lambda line: line)
    assert dataset[0] == "INFO 200"


def test_configurable_latin1_encoding(tmp_path: Path):
    path = tmp_path / "latin1.log"
    # 0xE9 is 'é' in latin-1; proves configured decoding is used.
    path.write_bytes(b"INFO caf\xe9\n")

    dataset = LogDataset(path, parser=lambda line: line, encoding="latin-1")
    assert dataset[0] == "INFO café"


def test_utf8_decode_error(tmp_path: Path):
    path = tmp_path / "bad-utf8.log"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(UnicodeDecodeError):
        LogDataset(path, parser=lambda line: line)


def test_blank_lines_preserved_by_default(tmp_path: Path):
    path = _write_log(tmp_path / "blank.log", "INFO 200\n\nERROR 500\n")
    dataset = LogDataset(path, parser=lambda line: line)
    assert len(dataset) == 3
    assert dataset[1] == ""


def test_skip_blank_lines_true(tmp_path: Path):
    path = _write_log(tmp_path / "blank.log", "INFO 200\n\n   \nERROR 500\n")
    dataset = LogDataset(path, parser=lambda line: line, skip_blank_lines=True)
    assert len(dataset) == 2
    assert dataset[0] == "INFO 200"
    assert dataset[1] == "ERROR 500"


def test_nonblank_whitespace_preserved(tmp_path: Path):
    path = _write_log(tmp_path / "ws.log", "   INFO 200   \n")
    seen: list[str] = []

    def parse(line: str):
        seen.append(line)
        return line

    dataset = LogDataset(path, parser=parse)
    assert dataset[0] == "   INFO 200   "
    assert seen == ["   INFO 200   "]


def test_physical_line_numbers_survive_blank_filtering(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "valid\n\nmalformed\n")

    def parse(line: str):
        if line == "malformed":
            raise ValueError("bad record")
        return line

    dataset = LogDataset(path, parser=parse, skip_blank_lines=True)
    assert len(dataset) == 2
    with pytest.raises(ValueError, match=r"line 3") as info:
        _ = dataset[1]
    assert "dataset index 1" in str(info.value)
    assert "server.log" in str(info.value)


def test_parser_error_includes_context(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "INFO 200\nBAD\n")

    def parse(line: str):
        if line == "BAD":
            raise KeyError("level")
        return line

    dataset = LogDataset(path, parser=parse)
    with pytest.raises(ValueError, match="Failed to parse log record") as info:
        _ = dataset[1]
    msg = str(info.value)
    assert "line 2" in msg
    assert "server.log" in msg
    assert "dataset index 1" in msg


def test_parser_error_chains_original_exception(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "BAD\n")

    def parse(line: str):
        raise KeyError("missing")

    dataset = LogDataset(path, parser=parse)
    with pytest.raises(ValueError) as info:
        _ = dataset[0]
    assert isinstance(info.value.__cause__, KeyError)
    assert info.value.__cause__.args == ("missing",)


def test_parser_errors_remain_lazy(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "ok\nBAD\n")

    def parse(line: str):
        if line == "BAD":
            raise ValueError("boom")
        return line

    dataset = LogDataset(path, parser=parse)
    assert len(dataset) == 2
    with pytest.raises(ValueError, match="Failed to parse"):
        _ = dataset[1]


def test_valid_lines_around_malformed_still_work(tmp_path: Path):
    path = _write_log(tmp_path / "server.log", "valid-a\nmalformed\nvalid-b\n")

    def parse(line: str):
        if line == "malformed":
            raise RuntimeError("nope")
        return line

    dataset = LogDataset(path, parser=parse)
    assert dataset[0] == "valid-a"
    with pytest.raises(ValueError, match="line 2"):
        _ = dataset[1]
    assert dataset[2] == "valid-b"


def test_crlf_handling(tmp_path: Path):
    path = tmp_path / "crlf.log"
    path.write_bytes(b"INFO 200\r\nERROR 500\r\n")
    seen: list[str] = []

    def parse(line: str):
        seen.append(line)
        return line

    dataset = LogDataset(path, parser=parse)
    assert seen == []  # lazy until access
    assert dataset[0] == "INFO 200"
    assert dataset[1] == "ERROR 500"
    assert all("\r" not in s and "\n" not in s for s in seen)


def test_all_blank_file_with_skipping(tmp_path: Path):
    path = _write_log(tmp_path / "blanks.log", "\n   \n\t\n")
    dataset = LogDataset(path, parser=lambda line: line, skip_blank_lines=True)
    assert len(dataset) == 0


def test_dataloader_with_skip_blank_lines(tmp_path: Path):
    path = _write_log(
        tmp_path / "server.log",
        "INFO 200 12\n\nERROR 500 83\nWARN 429 21\n",
    )

    def parse(line: str):
        level, status, latency = line.split()
        levels = {"INFO": 0.0, "WARN": 1.0, "ERROR": 2.0}
        return [levels[level], float(status), float(latency)]

    dataset = LogDataset(path, parser=parse, skip_blank_lines=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(dataset) == 3
    assert len(batches) == 2
    assert isinstance(batches[0], np.ndarray)
    assert batches[0].shape == (2, 3)
    assert np.allclose(batches[0], [[0.0, 200.0, 12.0], [2.0, 500.0, 83.0]])
    assert batches[1].shape == (1, 3)
