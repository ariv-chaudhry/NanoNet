"""Training history container with optional Matplotlib plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class History:
    """Stores per-epoch training and validation metrics.

    Attributes:
        loss: Training loss per epoch.
        accuracy: Training accuracy per epoch (if computed).
        val_loss: Validation loss per epoch.
        val_accuracy: Validation accuracy per epoch.
    """

    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)

    def update(self, **metrics: float | None) -> None:
        """Append metrics for the current epoch (``None`` values are skipped)."""
        mapping = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
        }
        for key, value in metrics.items():
            if value is None:
                continue
            if key not in mapping:
                raise KeyError(f"Unknown history metric: {key}")
            mapping[key].append(float(value))

    def plot(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> Any:
        """Plot loss (and accuracy if available) curves with Matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for History.plot(). Install with: pip install matplotlib"
            ) from exc

        has_acc = bool(self.accuracy or self.val_accuracy)
        n_plots = 2 if has_acc else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        epochs_loss = range(1, len(self.loss) + 1)
        axes[0].plot(epochs_loss, self.loss, label="train loss")
        if self.val_loss:
            axes[0].plot(range(1, len(self.val_loss) + 1), self.val_loss, label="val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

        if has_acc:
            if self.accuracy:
                axes[1].plot(range(1, len(self.accuracy) + 1), self.accuracy, label="train acc")
            if self.val_accuracy:
                axes[1].plot(
                    range(1, len(self.val_accuracy) + 1), self.val_accuracy, label="val acc"
                )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Accuracy")
            axes[1].legend()

        fig.tight_layout()
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        return fig
