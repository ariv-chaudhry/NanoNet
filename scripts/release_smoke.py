#!/usr/bin/env python3
"""Public-API smoke test for an installed NanoNet distribution.

Intended for clean wheel installs and the release workflow. Uses only the
public ``import nanonet_ml as nn`` surface.
"""

from __future__ import annotations

import numpy as np

import nanonet_ml as nn


def main() -> None:
    from importlib.metadata import version as pkg_version

    installed = pkg_version("nanonet-ml")
    assert nn.__version__ == installed, (nn.__version__, installed)

    nn.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = nn.Tensor(np.ones((3, 4)))
    output = model(x)
    assert output.shape == (3, 2)

    inspection = model.inspect(x, verbose=False)
    trace = model.trace(x, verbose=False)
    graph = output.graph(verbose=False)
    diagnostics = model.diagnose(x, verbose=False)

    assert inspection.total_parameters > 0
    assert len(trace.steps) == 3
    assert graph.root_id
    assert diagnostics is not None

    print(f"NanoNet {nn.__version__} release smoke test OK")
    print(f"  import file: {nn.__file__}")
    print(f"  parameters: {inspection.total_parameters}")
    print(f"  trace steps: {len(trace.steps)}")
    print(f"  graph ops: {len(graph.operations)}")
    print(f"  diagnose ok: {diagnostics.ok}")


if __name__ == "__main__":
    main()
