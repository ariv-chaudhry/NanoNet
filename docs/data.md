# Data loading in NanoNet

NanoNet provides a small, general data pipeline:

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
NanoNet Tensor / model / training
```

Concrete dataset implementations plug into the same `DataLoader`. NanoNet does
not automatically understand domain-specific file formats — for file-backed
sources such as logs, **you** define how each record becomes a sample.

## Responsibility boundaries

| Component | Responsibility |
| --- | --- |
| `Dataset` | Indexable sample source (`__len__`, `__getitem__`) |
| `TensorDataset` | Wrap aligned in-memory arrays / sequences |
| `LogDataset` | Read a text file, track physical lines, invoke a user parser |
| User parser | Semantic conversion of one line → features or `(features, target)` |
| `DataLoader` | Batching, shuffling, NumPy collation |
| Tensor / Trainer / model | Computation and training |

`DataLoader` currently returns **NumPy** batches. Convert to `nanonet_ml.Tensor`
at the point your training loop needs autograd — the same pattern used
internally by `Trainer`.

## Dataset protocol

`Dataset` is a structural protocol:

```python
from nanonet_ml.data import Dataset
```

Any object implementing `__len__` and `__getitem__` can be passed to
`DataLoader`. You do not need to subclass `Dataset`.

## TensorDataset

```python
from nanonet_ml.data import TensorDataset, DataLoader
import numpy as np

X = np.random.randn(100, 4)
y = np.random.randint(0, 2, size=100)
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True, seed=42)
```

## DataLoader

```python
from nanonet_ml.data import DataLoader
```

Supported behavior includes:

- mini-batching
- optional shuffling
- deterministic seeding via `seed=` or `nanonet_ml.manual_seed`
- `drop_last`
- tuple collation for supervised samples `(features, target)` → NumPy stacks

## LogDataset

`LogDataset` turns a line-oriented text file into a `Dataset` using a
**user-supplied parser**.

```python
from nanonet_ml.data import DataLoader, LogDataset


def parse_log(line: str):
    level, status, latency = line.split()
    levels = {"INFO": 0.0, "WARNING": 1.0, "ERROR": 2.0}
    return [levels[level], float(status), float(latency)]


dataset = LogDataset(
    "server.log",
    parser=parse_log,
    encoding="utf-8",
    skip_blank_lines=True,
)
```

### Constructor

```text
LogDataset(
    path,
    parser,
    *,
    encoding="utf-8",
    skip_blank_lines=False,
)
```

| Argument | Meaning |
| --- | --- |
| `path` | `str` or `pathlib.Path` to a text file (any extension) |
| `parser` | Callable receiving one logical line (no newline terminator) |
| `encoding` | Text encoding used to decode the file (default UTF-8) |
| `skip_blank_lines` | If `True`, omit blank / whitespace-only physical lines |

### Behavior notes

- One kept logical line → one dataset sample.
- Newline terminators (`\n`, `\r\n`) are removed; other whitespace on nonblank
  lines is preserved.
- Parsing is **lazy**: the parser runs in `__getitem__`, not at construction.
- Blank lines are **kept** by default (`skip_blank_lines=False`).
- Parser output may be feature-only or supervised `(features, target)`.
- Parser failures raise `ValueError` with file path, physical line number
  (1-based), and dataset index, chaining the original exception.
- NanoNet does **not** ship built-in Apache/NGINX/syslog parsers.

Design principle:

> NanoNet manages file-backed dataset mechanics.  
> The user determines what a log record means.

### End-to-end example

`examples/log_anomaly_detection.py` demonstrates a synthetic workflow:

```text
synthetic logs
    → parser
    → LogDataset
    → DataLoader
    → NanoNet MLP
    → anomaly classification
```

It is an educational demo, not production intrusion detection.

```bash
python examples/log_anomaly_detection.py
```

## MNIST utilities

```python
from nanonet_ml.data import download_mnist, load_mnist
```

See [`docs/mnist.md`](mnist.md) for preprocessing and the reference MLP example.

## Imports

```python
from nanonet_ml.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    LogDataset,
    load_mnist,
    download_mnist,
)
```
