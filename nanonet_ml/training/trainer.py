"""Reusable training loop."""

from __future__ import annotations

from typing import Any

import numpy as np

from nanonet.data.dataloader import DataLoader
from nanonet.data.dataset import TensorDataset
from nanonet.losses.cross_entropy import CrossEntropyLoss
from nanonet.metrics.accuracy import accuracy
from nanonet.nn.module import Module
from nanonet.optimizers.optimizer import Optimizer
from nanonet.tensor import Tensor, no_grad
from nanonet.training.history import History


class Trainer:
    """Orchestrates model training and evaluation.

    Prefer constructing a ``Trainer`` explicitly, or use ``model.fit(...)`` as
    a thin convenience wrapper around this class.
    """

    def __init__(self, model: Module) -> None:
        self.model = model

    def fit(
        self,
        X: np.ndarray | Tensor,
        y: np.ndarray | Tensor,
        *,
        loss_fn: Module,
        optimizer: Optimizer,
        epochs: int = 10,
        batch_size: int = 64,
        validation_data: tuple[Any, Any] | None = None,
        shuffle: bool = True,
        verbose: bool = True,
        compute_accuracy: bool | None = None,
    ) -> History:
        """Train ``self.model`` on ``(X, y)``.

        Args:
            X: Training features.
            y: Training targets / labels.
            loss_fn: Loss module.
            optimizer: Optimizer instance.
            epochs: Number of passes over the training set.
            batch_size: Mini-batch size.
            validation_data: Optional ``(X_val, y_val)``.
            shuffle: Shuffle training batches each epoch.
            verbose: Print per-epoch metrics.
            compute_accuracy: If None, auto-enable for CrossEntropyLoss.
        """
        if compute_accuracy is None:
            compute_accuracy = isinstance(loss_fn, CrossEntropyLoss)

        X_arr = X.data if isinstance(X, Tensor) else np.asarray(X)
        y_arr = y.data if isinstance(y, Tensor) else np.asarray(y)

        loader = DataLoader(
            TensorDataset(X_arr, y_arr),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        history = History()

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for X_batch, y_batch in loader:
                X_t = Tensor(X_batch)
                y_t = y_batch

                logits = self.model(X_t)
                loss = loss_fn(logits, y_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_n = len(X_batch)
                total_loss += float(loss.data) * batch_n
                total_examples += batch_n
                if compute_accuracy:
                    total_correct += (
                        accuracy(logits, y_batch) * batch_n
                    )

            train_loss = (
                total_loss / max(total_examples, 1)
            )
            train_acc = (
                total_correct / max(total_examples, 1)
                if compute_accuracy
                else None
            )

            val_loss = None
            val_acc = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_metrics = self._evaluate_raw(
                    X_val,
                    y_val,
                    loss_fn,
                    compute_accuracy,
                )
                val_loss = val_metrics["loss"]
                val_acc = val_metrics.get("accuracy")

            history.update(
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )

            if verbose:
                msg = f"Epoch {epoch}/{epochs}\n"
                msg += f"loss: {train_loss:.4f}"
                if train_acc is not None:
                    msg += (
                        f" - accuracy: {train_acc * 100:.2f}%"
                    )
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                if val_acc is not None:
                    msg += (
                        f" - val_accuracy: {val_acc * 100:.2f}%"
                    )
                print(msg)

        return history

    def evaluate(
        self,
        X: np.ndarray | Tensor,
        y: np.ndarray | Tensor,
        *,
        loss_fn: Module | None = None,
        batch_size: int = 256,
        compute_accuracy: bool = True,
    ) -> float | dict[str, float]:
        """Evaluate the model.

        If ``loss_fn`` is None and ``compute_accuracy`` is True, returns accuracy
        as a float (convenient for ``model.evaluate(X_test, y_test)``).
        Otherwise returns a metrics dictionary.
        """
        metrics = self._evaluate_raw(
            X,
            y,
            loss_fn,
            compute_accuracy,
            batch_size=batch_size,
        )
        if loss_fn is None and compute_accuracy:
            return metrics["accuracy"]
        return metrics

    def _evaluate_raw(
        self,
        X: Any,
        y: Any,
        loss_fn: Module | None,
        compute_accuracy: bool,
        batch_size: int = 256,
    ) -> dict[str, float]:
        self.model.eval()
        X_arr = (
            X.data if isinstance(X, Tensor) else np.asarray(X)
        )
        y_arr = (
            y.data if isinstance(y, Tensor) else np.asarray(y)
        )

        loader = DataLoader(
            TensorDataset(X_arr, y_arr),
            batch_size=batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0

        with no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(Tensor(X_batch))
                batch_n = len(X_batch)
                total_examples += batch_n

                if loss_fn is not None:
                    loss = loss_fn(logits, y_batch)
                    total_loss += float(loss.data) * batch_n

                if compute_accuracy:
                    total_correct += (
                        accuracy(logits, y_batch) * batch_n
                    )

        out: dict[str, float] = {}
        if loss_fn is not None:
            out["loss"] = (
                total_loss / max(total_examples, 1)
            )
        if compute_accuracy:
            out["accuracy"] = (
                total_correct / max(total_examples, 1)
            )
        return out