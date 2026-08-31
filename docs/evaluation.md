# Empirical Evaluation

## Purpose

This document reports a controlled comparison of **NanoNet** against **PyTorch**
on CPU. The goal is not to claim that NanoNet is competitive with production
frameworks on throughput. The goal is to show that NanoNet’s educational NumPy
implementation is **mathematically consistent** with an established reference
while quantifying the expected performance tradeoff.

We evaluate:

1. numerical parity (forward, loss, gradients, one SGD step)
2. multi-trial training and inference runtime
3. runtime scaling with dataset size
4. matched MNIST learning behaviour

All numbers below were produced by the scripts under `benchmarks/` on the
machine described in each JSON artifact under `results/`. Re-run the scripts
locally to regenerate values for your hardware.

## Experimental Environment

Recorded benchmark artifacts under `results/` were generated with NanoNet
**0.1.0** (see each JSON file's `environment` block). Those snapshots remain
the reference numbers documented below; v0.2.0 did not regenerate them.

Representative environment from
`results/numerical_parity.json`, `results/runtime_scaling.json`, and
`results/mnist_comparison.json`:

| Field | Value |
|-------|-------|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel64 Family 6 Model 170 (reported by `platform.processor()`) |
| Python | 3.13.14 |
| NanoNet | 0.1.0 |
| NumPy | 2.4.2 |
| PyTorch | 2.13.0+cpu |
| Device | CPU (both frameworks) |
| Dtype | float64 |

PyTorch may detect a GPU on some machines; the evaluation scripts still run the
primary comparison on **CPU** so the comparison isolates framework overhead
rather than accelerator differences.

Install evaluation dependencies with:

```bash
pip install -e ".[benchmark]"
```

---

## 1. Numerical Parity

### Method

Script: `python benchmarks/numerical_parity.py`

A tiny network is built in both frameworks:

```text
Linear(4, 5) -> ReLU -> Linear(5, 3) -> CrossEntropyLoss
```

Controls:

- one NumPy RNG seed generates inputs, labels, weights, and biases
- NanoNet weights use layout `(in_features, out_features)`
- PyTorch `Linear.weight` uses `(out_features, in_features)`; the transfer
  helper **transposes** explicitly
- batch size 8, float64, CPU
- mean-reduced cross-entropy on integer class indices (logits in, not softmax)
- plain SGD (`lr=0.01`, no momentum / weight decay)
- gradients cleared before each backward pass
- tolerances: `rtol=1e-7`, `atol=1e-9`

Compared quantities:

- forward logits
- scalar loss
- every parameter gradient
- parameters after one SGD update

### Results

Measured values from `results/numerical_parity.json`:

| Check | Metric | Value | Status |
|-------|--------|------:|:------:|
| Forward | max abs error | 2.22e-16 | PASS |
| Forward | mean abs error | 2.23e-17 | PASS |
| Loss | absolute difference | 2.22e-16 | PASS |
| Gradients | overall max abs error | 1.11e-16 | PASS |
| SGD update | overall max abs error | 8.67e-19 | PASS |
| **Overall** | | | **PASS** |

NanoNet loss ≈ 1.3283360756; PyTorch loss ≈ 1.3283360756.

### Interpretation

Under identical initialization and data, NanoNet and PyTorch agree at roughly
**machine-epsilon** scale for float64. Residual differences are consistent with
floating-point non-associativity and minor kernel/order differences—not with
incorrect autodiff or update rules.

---

## 2. Runtime Performance

### Method

Script: `python benchmarks/compare.py` (delegates to the scaling harness)

Workload:

- architecture `784 -> 128 -> 64 -> 10`
- Adam (`lr=1e-3`), synthetic random features/labels
- CPU, float64
- warm-up runs: 1
- measured runs: 5
- timings via `time.perf_counter()`
- model construction / imports excluded from timed sections where practical

### Results

From the `n=5000` entry in `results/runtime_scaling.json` (1 epoch, batch 64):

| Framework | Train mean (s) | Train std (s) | Infer mean (s / 256-batch) |
|-----------|---------------:|--------------:|---------------------------:|
| NanoNet | 7.34 | 0.81 | 0.194 |
| PyTorch | 2.37 | 0.77 | 0.020 |

Approximate slowdowns at `n=5000`:

- training ≈ **3.1×**
- inference ≈ **9.5×**

### Interpretation

NanoNet is slower, as expected. It emphasizes readable Python/NumPy graph
construction and educational clarity. PyTorch benefits from compiled kernels,
mature BLAS integration, and years of systems engineering. Slower NanoNet
runtime is treated as a **design tradeoff**, not a failed benchmark, unless a
specific algorithmic bug is identified.

---

## 3. Scaling Behaviour

### Method

Script: `python benchmarks/scaling_benchmark.py --sizes 1000 5000 10000 --runs 5`

Architecture and optimizer held fixed while dataset size varies. Training and
inference are timed separately (inference uses a fixed 256-example batch).

### Results

Training mean ± std (seconds):

| Samples | NanoNet | PyTorch | Train slowdown |
|--------:|--------:|--------:|---------------:|
| 1,000 | 1.38 ± 0.16 | 0.37 ± 0.07 | ~3.7× |
| 5,000 | 7.34 ± 0.81 | 2.37 ± 0.77 | ~3.1× |
| 10,000 | 12.97 ± 1.25 | 4.81 ± 1.83 | ~2.7× |

![NanoNet vs PyTorch runtime scaling](../results/runtime_scaling.png)

### Interpretation

Runtime increased with workload size in both implementations, while PyTorch
maintained substantially lower absolute execution time. A few discrete sizes do
not establish asymptotic complexity claims; they only illustrate the observed
trend on this hardware.

---

## 4. MNIST Learning Comparison

### Method

Script: `python benchmarks/mnist_comparison.py`

Matched controls:

- shared MNIST subset (5,000 train / 1,000 test by default)
- identical architecture `784 -> 128 -> 64 -> 10` (no Dropout)
- identical NumPy-initialized weights/biases copied into both models
- identical minibatch order (permutations generated once in NumPy)
- SGD (`lr=0.1`), mean cross-entropy, float64, CPU
- 1 epoch by default (CLI configurable)

### Results

From `results/mnist_comparison.json`:

| Framework | Test accuracy | Train time | Eval time | Epoch loss |
|-----------|--------------:|-----------:|----------:|-----------:|
| NanoNet | **66.50%** | 6.03 s | 0.027 s | 0.9144894894 |
| PyTorch | **66.50%** | 1.14 s | 0.016 s | 0.9144894894 |

Accuracy difference: **0.0**. Final loss difference ≈ **1.1e-16**.

### Interpretation

Under matched initialization, batch order, and hyperparameters, NanoNet and
PyTorch achieved **identical** predictive accuracy and essentially identical
loss after one epoch, while PyTorch completed training substantially faster.
This supports learning-behaviour equivalence for this dense MLP setting; it
does not by itself prove bit-identical computation for all future architectures.

Absolute accuracy after a single epoch on 5,000 samples is modest by design of
the default workload. Increase `--epochs` for stronger final accuracy if
desired.

---

## Limitations

- Benchmarks are **CPU-focused**; GPU acceleration is out of scope for NanoNet 0.1.
- Results depend on hardware, OS, and BLAS implementations used by NumPy/PyTorch.
- Small datasets can exaggerate Python/framework overhead relative to matmul work.
- Only dense MLP architectures are evaluated.
- Synthetic runtime workloads use random labels and measure performance, not quality.
- PyTorch exposes many optimizations and backends that NanoNet intentionally omits.
- These results are **not** claims about production deep-learning throughput.

---

## Reproducing the Experiments

```bash
pip install -e ".[benchmark]"

# Numerical correctness (exits non-zero on FAIL)
python benchmarks/numerical_parity.py

# Multi-trial runtime comparison
python benchmarks/compare.py --samples 5000 --runs 5

# Dataset-size scaling + plot
python benchmarks/scaling_benchmark.py --sizes 1000 5000 10000 --runs 5

# Matched MNIST learning comparison
python benchmarks/mnist_comparison.py --train-samples 5000 --test-samples 1000 --epochs 1

# Convenience wrappers
python benchmarks/run_evaluation.py --quick
python benchmarks/run_evaluation.py --full
```

Artifacts are written under `results/`:

- `numerical_parity.json`
- `runtime_scaling.json`
- `runtime_scaling.png`
- `mnist_comparison.json`
- `mnist_comparison.png`

Automated parity checks (skipped if PyTorch is absent):

```bash
pytest tests/test_numerical_parity.py -v
```
