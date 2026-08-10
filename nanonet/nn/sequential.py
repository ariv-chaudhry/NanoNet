"""Sequential container for stacking layers."""

from __future__ import annotations

from nanonet.nn.module import Module
from nanonet.tensor import Tensor


class Sequential(Module):
    """A sequence of modules applied in order.

    Accepts either a list/tuple of modules or modules as ``*args``::

        Sequential([Dense(10, 20), ReLU(), Dense(20, 2)])
        Sequential(Dense(10, 20), ReLU(), Dense(20, 2))
    """

    def __init__(self, *modules: Module | list[Module] | tuple[Module, ...]) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            layer_list = list(modules[0])
        else:
            layer_list = list(modules)

        for i, module in enumerate(layer_list):
            if not isinstance(module, Module):
                raise TypeError(
                    f"Sequential expected Module instances, got {type(module).__name__} at index {i}."
                )
            self.add_module(str(i), module)

    def add_module(self, name: str, module: Module) -> None:
        setattr(self, name, module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int | slice) -> Module | Sequential:
        layers = list(self._modules.values())
        if isinstance(index, slice):
            return Sequential(layers[index])
        return layers[index]
