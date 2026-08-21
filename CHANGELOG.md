# Changelog

All notable changes to NanoNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public package release candidate.

NanoNet is currently pre-1.0; public APIs may evolve as the framework matures.

### Added

- Tensor-based reverse-mode automatic differentiation
- Neural-network modules (`Module`, `Sequential`, `Parameter`)
- Layers including `Dense` / `Linear`, activations, `Dropout`, and `Flatten`
- Losses (`MSELoss`, `CrossEntropyLoss`) and optimizers (`SGD`, `Adam`)
- Data loading utilities and a lightweight trainer
- Model inspection (`model.inspect`)
- Execution tracing (`model.trace`)
- Autograd computation graph inspection (`tensor.graph`)
- Evidence-based model diagnostics (`model.diagnose`)
- Installable Python package with a curated top-level public API
