"""Base Module class for composable neural-network components."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np

from nanonet.nn.parameter import Parameter
from nanonet.tensor import Tensor, no_grad


class Module:
    """Base class for all neural-network modules.

    Subclasses implement ``forward``. Nested ``Module`` and ``Parameter``
    attributes are tracked automatically via ``__setattr__``.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # Remove stale registrations when overwriting an attribute.
        modules: OrderedDict[str, Module] = self.__dict__.get(
            "_modules",
            OrderedDict(),
        )
        params: OrderedDict[str, Parameter] = self.__dict__.get(
            "_parameters",
            OrderedDict(),
        )
        modules.pop(name, None)
        params.pop(name, None)

        if isinstance(value, Parameter):
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value

        object.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()."
        )

    def parameters(self) -> list[Parameter]:
        """Return all trainable parameters in this module (recursively)."""
        params: list[Parameter] = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(
        self,
        prefix: str = "",
    ) -> list[tuple[str, Parameter]]:
        """Return ``(name, parameter)`` pairs recursively."""
        result: list[tuple[str, Parameter]] = []
        for name, param in self._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            result.append((full, param))
        for name, module in self._modules.items():
            full = f"{prefix}.{name}" if prefix else name
            result.extend(module.named_parameters(full))
        return result

    def modules(self) -> Iterator[Module]:
        """Yield this module and all submodules depth-first."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> list[tuple[str, Module]]:
        """Return ``(hierarchical_name, module)`` for this module and descendants.

        The root entry uses ``prefix`` (empty string by default).
        """
        result: list[tuple[str, Module]] = [(prefix, self)]
        for name, module in self._modules.items():
            full = f"{prefix}.{name}" if prefix else name
            result.extend(module.named_modules(full))
        return result

    def train(self, mode: bool = True) -> Module:
        """Set training mode (affects Dropout and similar layers)."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Clear gradients on all parameters (sets ``grad`` to ``None``)."""
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self) -> OrderedDict[str, np.ndarray]:
        """Return a flat mapping of parameter names to NumPy arrays (copies)."""
        return OrderedDict(
            (name, param.data.copy())
            for name, param in self.named_parameters()
        )

    def load_state_dict(
        self,
        state_dict: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        """Load parameter values from a state dictionary.

        Args:
            state_dict: Mapping from parameter names to arrays.
            strict: If True, require exact key and shape matches.
        """
        current = dict(self.named_parameters())
        if strict:
            missing = set(current) - set(state_dict)
            unexpected = set(state_dict) - set(current)
            if missing or unexpected:
                raise KeyError(
                    f"state_dict mismatch. Missing keys: {sorted(missing)}; "
                    f"Unexpected keys: {sorted(unexpected)}"
                )
        for name, array in state_dict.items():
            if name not in current:
                if strict:
                    raise KeyError(
                        f"Unexpected key in state_dict: {name}"
                    )
                continue
            param = current[name]
            arr = np.asarray(array, dtype=param.data.dtype)
            if arr.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {param.shape}, got {arr.shape}."
                )
            param.data = arr.copy()

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count scalar parameters in the module."""
        total = 0
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            total += int(param.size)
        return total

    def summary(
        self,
        input_shape: tuple[int, ...] | None = None,
    ) -> str:
        """Return a human-readable summary of layers and parameter counts.

        Args:
            input_shape: Optional feature shape excluding batch, e.g. ``(784,)``.
                When provided, a dry-run forward pass estimates output shapes.
        """
        lines = [
            f"{'Layer':<28} {'Output Shape':<18} {'Parameters':>12}",
            "-" * 60,
        ]

        # Prefer Sequential-style ordered children; fall back to all submodules.
        children = list(self._modules.items())
        if not children:
            children = [(type(self).__name__, self)]

        shapes: dict[str, str] = {}
        if input_shape is not None:
            shapes = self._infer_shapes(input_shape)

        total = 0
        for name, module in children:
            n_params = module.num_parameters()
            total += n_params
            out_shape = shapes.get(name, "?")
            label = f"{type(module).__name__}"
            if (
                hasattr(module, "in_features")
                and hasattr(module, "out_features")
            ):
                label = (
                    f"{type(module).__name__}"
                    f"({module.in_features},{module.out_features})"
                )
            elif hasattr(module, "p"):
                label = f"{type(module).__name__}(p={module.p})"
            lines.append(
                f"{label:<28} {out_shape:<18} {n_params:>12}"
            )

        lines.append("-" * 60)
        lines.append(
            f"{'Total parameters:':<47} {total:>12}"
        )
        text = "\n".join(lines)
        print(text)
        return text

    def _infer_shapes(
        self,
        input_shape: tuple[int, ...],
    ) -> dict[str, str]:
        """Best-effort shape inference via a detached forward pass."""
        from nanonet.nn.sequential import Sequential

        shapes: dict[str, str] = {}
        if not isinstance(self, Sequential):
            return shapes

        x: Any = Tensor(
            np.zeros((1, *input_shape), dtype=np.float64),
            requires_grad=False,
        )
        was_training = self.training
        self.eval()
        try:
            with no_grad():
                for name, module in self._modules.items():
                    x = module(x)
                    if isinstance(x, Tensor):
                        shapes[name] = str(("?",) + x.shape[1:])
                    else:
                        shapes[name] = "?"
        except Exception:
            # Shape inference is best-effort; summary still shows parameter counts.
            pass
        finally:
            self.train(was_training)
        return shapes

    def save(self, path: str | Any) -> None:
        """Save parameters to an ``.npz`` file."""
        from nanonet.serialization import save_model

        save_model(self, path)

    def load(self, path: str | Any) -> None:
        """Load parameters from an ``.npz`` file."""
        from nanonet.serialization import load_model

        load_model(self, path)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience training wrapper; see ``nanonet.training.Trainer``."""
        from nanonet.training.trainer import Trainer

        return Trainer(self).fit(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience evaluation wrapper."""
        from nanonet.training.trainer import Trainer

        return Trainer(self).evaluate(*args, **kwargs)

    def inspect(
        self,
        x: Any | None = None,
        *,
        verbose: bool = True,
    ) -> Any:
        """Inspect model structure, and optionally runtime shapes / activations.

        Args:
            x: Optional sample input. When provided, runs a ``no_grad`` forward
                pass to capture per-layer shapes and activation statistics.
            verbose: If True (default), print a formatted report to stdout.

        Returns:
            A :class:`~nanonet.inspection.ModelInspectionReport`.
        """
        from nanonet.inspection import inspect_model

        return inspect_model(self, x, verbose=verbose)
