import os
import sys
from typing import Callable

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore


def oracle_f(x: float) -> tuple[float, float]:
    """
    A wrapper function around `oracle`.

    f(x), f'(x) = oracle_f(x)
    """
    return oracle(24233, x)


def gradient_descent(
    func: Callable[[float], tuple[float, float]],
    initial_x: float,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    max_iterations: int = int(1e6),
) -> float:
    """
    Minimizes a function f(x) using gradient descent.

    Solves the optimization problem:
    x* = argmin_x f(x)

    Parameters:
        func: Callable that takes a float x and returns a tuple (f(x), f'(x))
        initial_x: Starting point for gradient descent
        learning_rate: Step size (eta)
        tolerance: Stopping criterion for small gradients
        max_iterations: Maximum number of iterations

    Returns:
        x*: The value of x that (approximately) minimizes f(x)
    """
    x = initial_x
    for _ in range(max_iterations):
        fx, dfx = func(x)
        if abs(dfx) < tolerance:
            break
        x = x - learning_rate * dfx
    return x


if __name__ == "__main__":
    x_star = gradient_descent(
        func=oracle_f,
        initial_x=0.0,
        learning_rate=1e-3,
        tolerance=1e-6,
        max_iterations=int(1e6),
    )
    print(f"x* = {x_star:.6f}")
    fx_star, dfx_star = oracle_f(x_star)
    print(f"f(x*) = {fx_star:.6f}, f'(x*) = {dfx_star:.6f}")
    assert abs(dfx_star) < 1e-6, "Gradient at x* is not close to zero."
    print("Success! The gradient at x* is close to zero.")
