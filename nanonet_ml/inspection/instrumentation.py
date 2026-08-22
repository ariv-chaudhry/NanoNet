"""Temporary leaf-module forward instrumentation for observability features."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from nanonet_ml.inspection.utils import first_arg, leaf_modules
from nanonet_ml.nn.module import Module
from nanonet_ml.tensor import Tensor, no_grad


def unique_leaf_modules(model: Module) -> list[tuple[str, Module]]:
    """Leaf modules with hierarchical names, deduplicated by object identity.

    Shared modules appear once under their first discovered hierarchical name.
    """
    leaves = leaf_modules(model)
    unique: list[tuple[str, Module]] = []
    seen: set[int] = set()
    for name, mod in leaves:
        mid = id(mod)
        if mid in seen:
            continue
        seen.add(mid)
        unique.append((name, mod))
    return unique


@contextmanager
def instrument_leaf_forwards(
    model: Module,
    on_call: Callable[[str, Module, Any, Any, int], None],
) -> Iterator[None]:
    """Temporarily wrap unique leaf ``forward`` methods; always restore.

    ``on_call(layer_name, module, input_value, output, call_index)`` runs after
    each leaf forward. ``input_value`` is the first positional/keyword argument.
    Shared modules keep a rising ``call_index`` per hierarchical name.
    """
    unique_leaves = unique_leaf_modules(model)
    originals: dict[int, Callable[..., Any]] = {}
    had_instance_forward: dict[int, bool] = {}
    call_counts: dict[str, int] = defaultdict(int)

    for name, mod in unique_leaves:
        mid = id(mod)
        had_instance_forward[mid] = "forward" in mod.__dict__
        originals[mid] = mod.forward

        def _make_wrapper(
            layer_name: str,
            module_ref: Module,
            original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                inp = first_arg(args, kwargs)
                out = original(*args, **kwargs)
                call_counts[layer_name] += 1
                on_call(layer_name, module_ref, inp, out, call_counts[layer_name])
                return out

            return wrapped

        mod.forward = _make_wrapper(name, mod, originals[mid])  # type: ignore[method-assign]

    try:
        yield
    finally:
        for _name, mod in unique_leaves:
            mid = id(mod)
            if had_instance_forward[mid]:
                mod.forward = originals[mid]  # type: ignore[method-assign]
            elif "forward" in mod.__dict__:
                del mod.forward


def run_observed_forward(
    model: Module,
    x: Any,
    on_call: Callable[[str, Module, Any, Any, int], None],
    *,
    disable_grad: bool = True,
) -> Any:
    """Run ``model(x)`` once with temporary leaf instrumentation.

    Args:
        model: Root module.
        x: Input (wrapped as Tensor if needed).
        on_call: Invoked after each leaf forward as
            ``(name, module, input, output, call_index)``.
        disable_grad: If True, run under ``no_grad()`` (inspect/diagnose).
            If False, keep autograd enabled (trace).

    Returns:
        Model output. Underlying forward exceptions propagate after cleanup.
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)

    with instrument_leaf_forwards(model, on_call):
        if disable_grad:
            with no_grad():
                return model(x)
        return model(x)
