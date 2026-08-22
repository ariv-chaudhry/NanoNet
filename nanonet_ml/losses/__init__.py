"""Loss functions."""

from nanonet.losses.cross_entropy import CrossEntropyLoss
from nanonet.losses.mse import MSELoss

__all__ = ["MSELoss", "CrossEntropyLoss"]
