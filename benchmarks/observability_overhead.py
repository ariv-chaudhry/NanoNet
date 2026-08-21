"""Informational microbenchmark: normal forward vs observability APIs.

Observability calls intentionally collect metadata and are expected to be
slower than an unmodified forward. This script does not assert timings.
"""

from __future__ import annotations

import time

import numpy as np

from nanonet import Sequential, Tensor, manual_seed
from nanonet.layers import Dense, ReLU
from nanonet.losses import MSELoss


def _bench(label: str, fn, repeats: int = 50) -> float:
    # Warmup
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = (time.perf_counter() - t0) / repeats
    print(f"{label:28s}  {elapsed * 1e3:8.3f} ms / call")
    return elapsed


def main() -> None:
    manual_seed(0)
    model = Sequential(Dense(64, 128), ReLU(), Dense(128, 10))
    x = Tensor(np.random.randn(32, 64))
    y = Tensor(np.zeros((32, 10)))

    print("NanoNet observability overhead (informational)\n")
    _bench("normal forward", lambda: model(x))
    _bench("inspect(x)", lambda: model.inspect(x, verbose=False))
    _bench("trace(x)", lambda: model.trace(x, verbose=False))
    _bench("diagnose(x)", lambda: model.diagnose(x, verbose=False))

    pred = model(x)
    loss = MSELoss()(pred, y)
    _bench("loss.graph()", lambda: loss.graph(verbose=False))

    print(
        "\nNote: normal forward should not allocate observability records. "
        "Observability APIs are allowed to be slower."
    )


if __name__ == "__main__":
    main()
