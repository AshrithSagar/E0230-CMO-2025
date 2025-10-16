"""
src/functions.py
=======
Optimisation test problems

References
-------
- https://www.sfu.ca/~ssurjano/optimization.html
"""

import numpy as np

from .types import floatVec


def rosenbrock(x: floatVec, a: float = 1.0, b: float = 100.0) -> float:
    """The Rosenbrock function"""
    return sum((a - x[:-1]) ** 2.0 + b * (x[1:] - x[:-1] ** 2.0) ** 2.0)


def rosenbrock_grad(x: floatVec, a: float = 1.0, b: float = 100.0) -> floatVec:
    """Gradient of the Rosenbrock function"""
    grad = np.zeros_like(x)
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    for i in range(1, len(x) - 1):
        grad[i] = (
            2 * (x[i] - a)
            - 4 * b * x[i] * (x[i + 1] - x[i] ** 2)
            + 2 * b * (x[i] - x[i - 1] ** 2)
        )
    grad[-1] = 2 * b * (x[-1] - x[-2] ** 2)
    return grad


def rastrigin(x: floatVec, A: float = 10.0) -> float:
    """The Rastrigin function"""
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))
