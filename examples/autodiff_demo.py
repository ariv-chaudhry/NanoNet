"""Demonstrate NanoNet automatic differentiation."""

from nanonet import Tensor


def main() -> None:
    print("=== Scalar autodiff ===")
    print("y = x^2 + 2x")
    print("dy/dx = 2x + 2")
    print("At x = 3: dy/dx = 8\n")

    x = Tensor(3.0, requires_grad=True)
    y = x**2 + 2 * x
    y.backward()
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"x.grad = {x.grad}  (expected 8.0)\n")

    print("=== Multivariable example ===")
    print("y = x*w + b")
    print("dy/dx = w, dy/dw = x, dy/db = 1\n")

    x = Tensor(2.0, requires_grad=True)
    w = Tensor(3.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)
    y = x * w + b
    y.backward()
    print(f"x.grad = {x.grad}  (expected 3.0)")
    print(f"w.grad = {w.grad}  (expected 2.0)")
    print(f"b.grad = {b.grad}  (expected 1.0)\n")

    print("=== Branching graph ===")
    print("y = x*x + 3*x  => dy/dx = 2x + 3")
    print("At x = 2: dy/dx = 7\n")

    x = Tensor(2.0, requires_grad=True)
    y = x * x + 3 * x
    y.backward()
    print(f"x.grad = {x.grad}  (expected 7.0)")


if __name__ == "__main__":
    main()
