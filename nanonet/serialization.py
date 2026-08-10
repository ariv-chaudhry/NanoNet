"""Model serialization using NumPy ``.npz`` (no pickle)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nanonet.nn.module import Module


def save_model(model: Module, path: str | Path) -> None:
    """Save model parameters to ``path`` (``.npz``).

    Arrays are stored under their ``state_dict`` keys. A small JSON sidecar
    ``path.with_suffix('.json')`` records metadata (parameter names / shapes).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    # np.savez cannot use keys with some characters; replace carefully.
    # Keep original names via a metadata list.
    arrays = {f"p{i}": arr for i, arr in enumerate(state.values())}
    np.savez(path, **arrays)

    meta = {
        "format": "nanonet-npz-v1",
        "keys": list(state.keys()),
        "shapes": {k: list(v.shape) for k, v in state.items()},
    }
    meta_path = path.with_suffix(path.suffix + ".json") if path.suffix else Path(str(path) + ".json")
    # Prefer model.npz.json style next to file
    meta_path = Path(str(path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(model: Module, path: str | Path) -> Module:
    """Load parameters from ``path`` into ``model`` in-place and return it."""
    path = Path(path)
    meta_path = Path(str(path) + ".meta.json")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {meta_path}. "
            "Re-save the model with nanonet.serialization.save_model."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys: list[str] = meta["keys"]
    with np.load(path) as data:
        state = {keys[i]: data[f"p{i}"] for i in range(len(keys))}
    model.load_state_dict(state)
    return model
