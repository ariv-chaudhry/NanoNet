"""Numerically stable cross-entropy loss for classification."""

from __future__ import annotations

import numpy as np

from nanonet.nn.module import Module
from nanonet.tensor import Function, Tensor


class CrossEntropyFunction(Function):
    """Softmax + NLL fused for numerical stability (log-sum-exp)."""

    def _forward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if logits.ndim != 2:
            raise ValueError(
                f"CrossEntropyLoss expected 2D logits (batch, classes), got shape {logits.shape}."
            )
        batch, _ = logits.shape
        # Preserve user-facing shape checks before flattening.
        raw = np.asarray(targets)
        if raw.ndim != 1:
            raise ValueError(
                f"CrossEntropyLoss expected integer labels with shape ({batch},), "
                f"received shape {raw.shape}."
            )
        targets = raw.astype(np.int64).reshape(-1)
        if targets.shape != (batch,):
            raise ValueError(
                f"CrossEntropyLoss expected integer labels with shape ({batch},), "
                f"received shape {targets.shape}."
            )
        if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
            raise ValueError(
                f"CrossEntropyLoss labels must be in [0, {logits.shape[1] - 1}], "
                f"got min={targets.min()}, max={targets.max()}."
            )

        # log_softmax via log-sum-exp: logsumexp = m + log(sum(exp(x-m)))
        m = np.max(logits, axis=1, keepdims=True)
        logsumexp = m + np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))
        log_probs = logits - logsumexp

        self._log_probs = log_probs
        self._targets = targets
        self._batch = batch

        nll = -log_probs[np.arange(batch), targets]
        return np.asarray(nll.mean(), dtype=np.float64)

    def _backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        # Softmax probabilities: exp(log_probs)
        probs = np.exp(self._log_probs)
        grad_logits = probs
        grad_logits[np.arange(self._batch), self._targets] -= 1.0
        grad_logits /= self._batch
        grad_logits *= grad_output
        # Targets are discrete labels — no gradient.
        return (grad_logits if self.needs_input_grad[0] else None, None)


class CrossEntropyLoss(Module):
    """Cross-entropy loss from raw logits and integer class labels.

    Combines softmax and negative log-likelihood. Do **not** apply Softmax
    before this loss.

    Args used in forward:
        logits: Shape ``(batch, num_classes)``.
        targets: Integer labels of shape ``(batch,)``.
    """

    def forward(self, logits: Tensor, targets: Tensor | object) -> Tensor:
        if not isinstance(logits, Tensor):
            logits = Tensor(logits)
        if isinstance(targets, Tensor):
            target_data = targets.data
        else:
            target_data = np.asarray(targets)

        # Pass targets as a non-grad Tensor so Function.apply accepts them.
        target_tensor = Tensor(target_data.astype(np.float64), requires_grad=False)
        return CrossEntropyFunction().apply(logits, target_tensor)
