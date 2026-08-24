# NanoNet 0.1.0

NanoNet 0.1.0 is the first public release of NanoNet, a lightweight
neural-network framework built from scratch with Python and NumPy and designed
around transparent internals and built-in observability.

NanoNet is not intended to replace PyTorch. Its goal is to make the mechanics
behind neural-network training easier to inspect, understand, and debug.

## Highlights

- Tensor operations with reverse-mode automatic differentiation
- Neural-network modules, layers, activations, losses, and optimizers
- Data loading and reusable training utilities
- Model inspection with `model.inspect()`
- Forward execution tracing with `model.trace(x)`
- Autograd computation graph inspection with `tensor.graph()`
- Model diagnostics with `model.diagnose(x)`
- Serialization and state-management utilities
- Empirical benchmarking and numerical comparisons against PyTorch
- Public Python package distribution through PyPI

## Installation

```bash
pip install nanonet-ml
```

```python
import nanonet_ml as nn
```

## Version

```text
0.1.0
```

NanoNet remains pre-1.0, so the public API may continue to evolve as the
framework develops.

## License

NanoNet is source-available and is not distributed under an open-source
license. See `LICENSE` for the complete terms.