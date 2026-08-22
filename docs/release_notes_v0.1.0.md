# NanoNet 0.1.0

NanoNet's first public release introduces a lightweight neural-network framework
with built-in observability.

NanoNet is designed around transparent internals — not as a replacement for
PyTorch.

## Highlights

- automatic differentiation
- neural-network modules and layers
- model inspection with `model.inspect()`
- execution tracing with `model.trace(x)`
- computation graph inspection with `tensor.graph()`
- diagnostics with `model.diagnose(x)`
- benchmarking against reference frameworks
- installable package: `pip install nanonet-ml`

## Installation

```bash
pip install nanonet-ml
```

```python
import nanonet_ml as nn
```

## License

Source-available / proprietary — see `LICENSE`. Viewing and educational
evaluation are permitted; redistribution requires permission.
