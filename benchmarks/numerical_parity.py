"""Numerical parity: NanoNet vs PyTorch on identical inputs and parameters.

Compares forward logits, cross-entropy loss, parameter gradients, and one
plain SGD update using synchronized float64 tensors on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import (  # noqa: E402
    RESULTS_DIR,
    assign_numpy_params_to_nanonet,
    compare_arrays,
    copy_nanonet_weights_to_pytorch,
    environment_metadata,
    extract_nanonet_grads,
    extract_nanonet_params,
    extract_pytorch_grads,
    extract_pytorch_params,
    nanonet_mlp,
    pytorch_mlp,
    require_torch,
    save_json,
    set_global_seeds,
)


def _build_shared_params(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create NanoNet-layout (in, out) weights and biases for 4→5→3."""
    w1 = rng.normal(0.0, 0.5, size=(4, 5)).astype(np.float64)
    b1 = rng.normal(0.0, 0.1, size=(5,)).astype(np.float64)
    w2 = rng.normal(0.0, 0.5, size=(5, 3)).astype(np.float64)
    b2 = rng.normal(0.0, 0.1, size=(3,)).astype(np.float64)
    return [(w1, b1), (w2, b2)]


def run_parity(
    *,
    seed: int = 0,
    lr: float = 0.01,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    out: Path | None = None,
) -> dict:
    """Execute the full numerical-parity experiment and return a result dict."""
    torch = require_torch()
    import torch.nn as nn

    from nanonet import Tensor
    from nanonet.losses import CrossEntropyLoss
    from nanonet.optimizers import SGD

    set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    sizes = (4, 5, 3)
    batch = 8
    x_np = rng.normal(0.0, 1.0, size=(batch, sizes[0])).astype(np.float64)
    y_np = rng.integers(0, sizes[-1], size=(batch,), dtype=np.int64)
    params = _build_shared_params(rng)

    # NanoNet forward + backward + SGD
    nn_model = nanonet_mlp(sizes)
    assign_numpy_params_to_nanonet(nn_model, params)
    nn_opt = SGD(nn_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    nn_loss_fn = CrossEntropyLoss()

    nn_model.train()
    nn_opt.zero_grad()
    nn_logits = nn_model(Tensor(x_np))
    nn_loss = nn_loss_fn(nn_logits, y_np)
    nn_loss.backward()

    nn_logits_np = np.asarray(nn_logits.data, dtype=np.float64)
    nn_loss_val = float(nn_loss.data)
    nn_grads = extract_nanonet_grads(nn_model)
    nn_opt.step()
    nn_params_after = extract_nanonet_params(nn_model)

    # PyTorch with identical parameters (weight transpose handled in helper)
    torch.set_default_dtype(torch.float64)
    pt_model = pytorch_mlp(sizes, dtype=torch.float64)
    src = nanonet_mlp(sizes)
    assign_numpy_params_to_nanonet(src, params)
    copy_nanonet_weights_to_pytorch(src, pt_model)

    pt_opt = torch.optim.SGD(pt_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    pt_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)

    pt_model.train()
    pt_opt.zero_grad()
    pt_logits = pt_model(x_t)
    pt_loss = pt_loss_fn(pt_logits, y_t)
    pt_loss.backward()

    pt_logits_np = pt_logits.detach().cpu().numpy().astype(np.float64)
    pt_loss_val = float(pt_loss.detach().cpu().item())
    pt_grads = extract_pytorch_grads(pt_model)
    pt_opt.step()
    pt_params_after = extract_pytorch_params(pt_model)

    forward = compare_arrays(nn_logits_np, pt_logits_np, rtol=rtol, atol=atol, name="logits")
    loss_abs = abs(nn_loss_val - pt_loss_val)
    loss_pass = bool(np.isclose(nn_loss_val, pt_loss_val, rtol=rtol, atol=atol))

    grad_rows = []
    overall_grad_max = 0.0
    grads_pass = True
    for (n_name, n_g), (p_name, p_g) in zip(nn_grads, pt_grads):
        if n_name != p_name:
            raise RuntimeError(f"Gradient name mismatch: {n_name} vs {p_name}")
        row = compare_arrays(n_g, p_g, rtol=rtol, atol=atol, name=n_name)
        grad_rows.append(row)
        overall_grad_max = max(overall_grad_max, row["max_abs_error"])
        grads_pass = grads_pass and row["allclose"]

    param_rows = []
    overall_param_max = 0.0
    params_pass = True
    for (n_name, n_p), (p_name, p_p) in zip(nn_params_after, pt_params_after):
        if n_name != p_name:
            raise RuntimeError(f"Parameter name mismatch: {n_name} vs {p_name}")
        row = compare_arrays(n_p, p_p, rtol=rtol, atol=atol, name=n_name)
        param_rows.append(row)
        overall_param_max = max(overall_param_max, row["max_abs_error"])
        params_pass = params_pass and row["allclose"]

    overall = bool(forward["allclose"] and loss_pass and grads_pass and params_pass)

    result = {
        "experiment": "numerical_parity",
        "environment": environment_metadata(
            dtype="float64",
            device="CPU",
            seed=seed,
            extra={
                "architecture": "4 → 5 → ReLU → 3",
                "batch_size": batch,
                "loss": "CrossEntropyLoss(mean)",
                "optimizer": f"SGD(lr={lr})",
                "rtol": rtol,
                "atol": atol,
            },
        ),
        "configuration": {
            "seed": seed,
            "lr": lr,
            "rtol": rtol,
            "atol": atol,
            "batch_size": batch,
            "input_features": sizes[0],
            "classes": sizes[-1],
        },
        "results": {
            "forward": forward,
            "loss": {
                "nanonet": nn_loss_val,
                "pytorch": pt_loss_val,
                "absolute_difference": loss_abs,
                "allclose": loss_pass,
            },
            "gradients": {
                "per_parameter": grad_rows,
                "overall_max_abs_error": overall_grad_max,
                "allclose": grads_pass,
            },
            "sgd_update": {
                "per_parameter": param_rows,
                "overall_max_abs_error": overall_param_max,
                "allclose": params_pass,
            },
            "overall_pass": overall,
        },
    }

    out_path = Path(out) if out is not None else RESULTS_DIR / "numerical_parity.json"
    save_json(out_path, result)
    result["_out_path"] = str(out_path)
    return result


def _fmt(x: float) -> str:
    return f"{x:.6e}"


def print_report(result: dict) -> None:
    r = result["results"]
    print("NanoNet vs PyTorch Numerical Parity")
    print("=" * 35)
    print()
    print("Forward Pass")
    print("------------")
    print(f"Max absolute error:  {_fmt(r['forward']['max_abs_error'])}")
    print(f"Mean absolute error: {_fmt(r['forward']['mean_abs_error'])}")
    print("PASS" if r["forward"]["allclose"] else "FAIL")
    print()
    print("Loss")
    print("----")
    print(f"NanoNet: {r['loss']['nanonet']:.10f}")
    print(f"PyTorch: {r['loss']['pytorch']:.10f}")
    print(f"Absolute difference: {_fmt(r['loss']['absolute_difference'])}")
    print("PASS" if r["loss"]["allclose"] else "FAIL")
    print()
    print("Gradients")
    print("---------")
    for row in r["gradients"]["per_parameter"]:
        status = "PASS" if row["allclose"] else "FAIL"
        print(f"{row['name']:<16} {_fmt(row['max_abs_error'])}   {status}")
    print(f"Overall max abs:   {_fmt(r['gradients']['overall_max_abs_error'])}")
    print()
    print("SGD Update")
    print("----------")
    print(f"Max parameter difference: {_fmt(r['sgd_update']['overall_max_abs_error'])}")
    print("PASS" if r["sgd_update"]["allclose"] else "FAIL")
    print()
    print(f"Overall Result: {'PASS' if r['overall_pass'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoNet vs PyTorch numerical parity check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "numerical_parity.json")
    args = parser.parse_args()

    result = run_parity(seed=args.seed, lr=args.lr, rtol=args.rtol, atol=args.atol, out=args.out)
    print_report(result)
    print(f"\nSaved: {result['_out_path']}")
    raise SystemExit(0 if result["results"]["overall_pass"] else 1)


if __name__ == "__main__":
    main()
