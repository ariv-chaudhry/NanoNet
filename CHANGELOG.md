# Changelog

All notable changes to NanoNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-23

First public release of NanoNet.

NanoNet is a lightweight neural-network framework built from scratch in Python
with transparent automatic differentiation, training utilities, and built-in
model observability.

### Added

- Tensor operations with reverse-mode automatic differentiation
- Neural-network modules including `Module`, `Parameter`, and `Sequential`
- `Dense` / `Linear` layers
- Activation functions including `ReLU`, `Sigmoid`, `Tanh`, and `Softmax`
- `Dropout` and `Flatten`
- Loss functions including `MSELoss` and `CrossEntropyLoss`
- Optimizers including `SGD` and `Adam`
- Data loading and training utilities
- Model serialization and state management
- Model inspection with `model.inspect()`
- Forward execution tracing with `model.trace(...)`
- Autograd computation graph inspection with `tensor.graph()`
- Model diagnostics with `model.diagnose(...)`
- Empirical benchmarking and numerical comparisons against PyTorch
- Public package distribution through PyPI
- Support for installation with `pip install nanonet-ml`
- Public import namespace `nanonet_ml`
- Versioned package API with `nanonet_ml.__version__`

### Changed

- Renamed the Python package namespace from `nanonet` to `nanonet_ml`
- Set the PyPI distribution name to `nanonet-ml`
- Updated tests, examples, benchmarks, documentation, and CI to use the new namespace
- Updated packaging and release infrastructure for the first public release

### Notes

NanoNet remains a pre-1.0 project. Public APIs may continue to evolve as the
framework develops.