# MNIST with NanoNet

## Dataset

MNIST contains 70,000 handwritten digit images (0–9):

* 60,000 training samples
* 10,000 test samples
* 28×28 grayscale pixels

NanoNet downloads the original IDX `.gz` files into `data/mnist/` (gitignored)
via `nanonet_ml.data.load_mnist` or `scripts/download_mnist.py`.

## Preprocessing

1. Flatten each image to a vector of length 784
2. Scale pixel values to `[0, 1]` by dividing by 255
3. Keep labels as integer class indices

## Architecture

The reference MLP in `examples/mnist_mlp.py`:

```text
Input (784)
    │
    ▼
Dense (128)
    │
    ▼
ReLU
    │
    ▼
Dropout (optional)
    │
    ▼
Dense (64)
    │
    ▼
ReLU
    │
    ▼
Dense (10)
    │
    ▼
Cross Entropy (on logits)
```

## Training setup

| Item | Default |
|------|---------|
| Optimizer | Adam (`lr=1e-3`) |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Dropout | 0.2 |
| Epochs | 5 (CLI configurable) |

## Reproduce results

Full training:

```bash
python examples/mnist_mlp.py --epochs 10 --batch-size 64 --lr 0.001
```

Fast smoke test:

```bash
python examples/mnist_mlp.py --epochs 1 --train-limit 5000
```

### Measured smoke-test result

On one development run:

```bash
python examples/mnist_mlp.py --epochs 1 --train-limit 5000
```

produced approximately **88% test accuracy** after a single epoch on 5,000
training samples (seed 42). Full training with more epochs typically reaches
the mid/high 90% range — re-run locally and use the printed metric.

## Notes

* CI does **not** download MNIST or run full training.
* If the network is unavailable, `load_mnist` raises a clear error describing
  where to place the IDX files manually.
