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


def gradient_descent_momentum(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-3,
    momentum: float = 0.9,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    x = x0
    v = 0.0  # Velocity
    for _ in range(maxiter):
        _, grad = func(x)
        if abs(grad) < tol:
            break
        v = momentum * v - lr * grad
        x += v
    return x


def nesterov_accelerated_gradient(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-3,
    momentum: float = 0.9,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    x = x0
    v = 0.0
    for _ in range(maxiter):
        # Look-ahead point
        _, grad = func(x + momentum * v)
        if abs(grad) < tol:
            break
        v = momentum * v - lr * grad
        x += v
    return x


def adagrad(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-2,
    eps: float = 1e-8,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    x = x0
    grad_squared_sum = 0.0
    for _ in range(maxiter):
        _, grad = func(x)
        if abs(grad) < tol:
            break
        grad_squared_sum += grad**2
        adjusted_lr = lr / (grad_squared_sum**0.5 + eps)
        x -= adjusted_lr * grad
    return x


def rmsprop(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-3,
    beta: float = 0.9,
    eps: float = 1e-8,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    x = x0
    avg_sq_grad = 0.0
    for _ in range(maxiter):
        _, grad = func(x)
        if abs(grad) < tol:
            break
        avg_sq_grad = beta * avg_sq_grad + (1 - beta) * grad**2
        x -= lr * grad / (avg_sq_grad**0.5 + eps)
    return x


def adam(
    func: FirstOrderOracleFn,
    x0: float,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-6,
    maxiter: int = int(1e6),
) -> float:
    x = x0
    m = 0.0  # First moment
    v = 0.0  # Second moment
    for t in range(1, maxiter + 1):
        _, grad = func(x)
        if abs(grad) < tol:
            break
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (v_hat**0.5 + eps)
    return x


if __name__ == "__main__":
    methods = {
        "Gradient Descent": gradient_descent,
        "Momentum": gradient_descent_momentum,
        "NAG": nesterov_accelerated_gradient,
        "Adagrad": adagrad,
        "RMSProp": rmsprop,
        "Adam": adam,
    }
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        x_star = method(func=oracle_f, x0=0.0)
        fx_star, dfx_star = oracle_f(x_star)
        print(f"x* = {x_star:.6f}, f(x*) = {fx_star:.6f}, f'(x*) = {dfx_star:.6e}")
        if abs(dfx_star) < 1e-6:
            print("Success! Gradient is close to zero.")
        else:
            print("Error: Gradient is not close to zero.")
    print("\nAll methods converged successfully!")
