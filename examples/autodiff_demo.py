"""Demonstrate NanoNet automatic differentiation."""

import nanonet_ml as nn


def main() -> None:
    print("=== Scalar autodiff ===")
    print("y = x^2 + 2x")
    print("dy/dx = 2x + 2")
    print("At x = 3: dy/dx = 8\n")

    x = nn.Tensor(3.0, requires_grad=True)
    y = x**2 + 2 * x
    y.backward()
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"x.grad = {x.grad}  (expected 8.0)\n")

    print("=== Multivariable example ===")
    print("y = x*w + b")
    print("At x=2, w=3, b=1: y=7, dy/dx=3, dy/dw=2, dy/db=1\n")

    x = nn.Tensor(2.0, requires_grad=True)
    w = nn.Tensor(3.0, requires_grad=True)
    b = nn.Tensor(1.0, requires_grad=True)
    y = x * w + b
    y.backward()
    print(f"y = {y.data}")
    print(f"x.grad = {x.grad}")
    print(f"w.grad = {w.grad}")
    print(f"b.grad = {b.grad}")


if __name__ == "__main__":
    main()
