"""Classification accuracy metric."""

from __future__ import annotations

import numpy as np

from nanonet.tensor import Tensor


def accuracy(logits: Tensor | np.ndarray, targets: Tensor | np.ndarray) -> float:
    """Compute classification accuracy from logits and integer labels.

    Predicted class is ``argmax(logits, axis=1)``.

    Args:
        logits: Array of shape ``(batch, num_classes)``.
        targets: Integer labels of shape ``(batch,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if isinstance(logits, Tensor):
        logits_arr = logits.data
    else:
        logits_arr = np.asarray(logits)
    if isinstance(targets, Tensor):
        targets_arr = targets.data
    else:
        targets_arr = np.asarray(targets)

    if logits_arr.ndim != 2:
        raise ValueError(f"accuracy expected 2D logits, got shape {logits_arr.shape}.")
    preds = np.argmax(logits_arr, axis=1)
    targets_arr = targets_arr.astype(np.int64).reshape(-1)
    if preds.shape != targets_arr.shape:
        raise ValueError(
            f"accuracy shape mismatch: preds {preds.shape} vs targets {targets_arr.shape}."
        )
    if len(preds) == 0:
        return 0.0
    return float(np.mean(preds == targets_arr))
