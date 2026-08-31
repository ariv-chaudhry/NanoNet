# NanoNet 0.2.0

NanoNet 0.2.0 adds parser-driven, file-backed log ingestion on top of the
existing Dataset / DataLoader architecture.

This release does not change Tensor, Module, Trainer, or optimizer behavior.
`LogDataset` is a new dataset implementation that plugs into the pipeline
NanoNet already had.

## Highlights

- `LogDataset` for line-oriented text / log files
- User-supplied parsers for feature-only or supervised samples
- Configurable encoding and optional blank-line filtering
- Physical source-line tracking with contextual parser errors
- End-to-end example: `examples/log_anomaly_detection.py`
- Documentation for the data pipeline in `docs/data.md`

## Installation

```bash
pip install --upgrade nanonet-ml
```

```python
import nanonet_ml as nn
from nanonet_ml.data import DataLoader, LogDataset
```

## Version

```text
0.2.0
```

## License

NanoNet is source-available and is not distributed under an open-source
license. See `LICENSE` for the complete terms.
