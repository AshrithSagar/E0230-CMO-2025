import os
import sys
from typing import Callable, TypeAlias

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

SRN = 24233
"""
The 5-digit Student Registration Number (SRN) for the assignment.
"""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."


def oracle_f(x: float) -> tuple[float, float]:
    """
    A wrapper around the provided `oracle` function.

    `f(x), f'(x) = oracle_f(x)`
    """
    return oracle(SRN, x)


FirstOrderOracleFn: TypeAlias = Callable[[float], tuple[float, float]]
"""
Type alias for a first-order oracle function.
It takes a float `x` and returns a tuple `(f(x), f'(x))`.
"""


def gradient_descent(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-3,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    """
    Minimizes a function `f(x)` using gradient descent.

    Solves the optimization problem:
    `x* = argmin_x f(x)`

    Parameters:
        func: Callable that takes a float `x` and returns a tuple `(f(x), f'(x))`.
        x0: Initial guess for the minimum point.
        lr: Learning rate / Step size (`eta`).
        tol: Tolerance, stopping criterion for small gradients.
        maxiter: Maximum number of iterations to perform.

    Returns:
        x*: The value of `x` that (approximately) minimizes `f(x)`
    """
    x = x0
    for _ in range(maxiter):
        fx, dfx = func(x)  # Query the oracle fn
        if abs(dfx) < tol:
            break  # Early exit
        x = x - lr * dfx  # Gradient descent step
    return x


if __name__ == "__main__":
    x_star = gradient_descent(
        func=oracle_f,
        x0=0.0,
        lr=1e-3,
        tol=1e-6,
        maxiter=int(1e6),
    )
    print(f"x* = {x_star:.6f}")
    fx_star, dfx_star = oracle_f(x_star)
    print(f"f(x*) = {fx_star:.6f}, f'(x*) = {dfx_star:.6f}")
    assert abs(dfx_star) < 1e-6, "Gradient at x* is not close to zero."
    print("Success! The gradient at x* is close to zero.")
