# Changelog

All notable changes to NanoNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release engineering: PyPI Trusted Publishing workflow, `RELEASING.md`, and
  release smoke / tag-consistency scripts

### Changed

- Python import package renamed to `nanonet_ml` and PyPI distribution name set to
  `nanonet-ml` to avoid colliding with an existing `nanonet` package on PyPI.
  Project branding remains **NanoNet**. Recommended usage:
  `import nanonet_ml as nn`.

## [0.1.0] - Unreleased

First public package release.

NanoNet is a lightweight neural-network framework designed around transparent
internals and built-in observability. It is pre-1.0; public APIs may evolve.

Set the release date (YYYY-MM-DD) in the release-preparation commit before
tagging `v0.1.0`. See `RELEASING.md`.

### Added

- Tensor-based reverse-mode automatic differentiation
- Neural-network modules (`Module`, `Sequential`, `Parameter`)
- Layers including `Dense` / `Linear`, activations, `Dropout`, and `Flatten`
- Losses (`MSELoss`, `CrossEntropyLoss`) and optimizers (`SGD`, `Adam`)
- Data loading utilities and a lightweight trainer
- Model inspection with `model.inspect()`
- Forward execution tracing with `model.trace(x)`
- Autograd computation graph inspection with `tensor.graph()`
- Evidence-based model diagnostics with `model.diagnose()`
- Benchmarking / comparison infrastructure against reference frameworks
- Installable public package API (`import nanonet_ml as nn`)
- PyPI distribution `nanonet-ml` (import package `nanonet_ml`)
