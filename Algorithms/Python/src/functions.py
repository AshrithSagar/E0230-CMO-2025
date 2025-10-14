"""
functions.py
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


def rastrigin(x: floatVec, A: float = 10.0) -> float:
    """The Rastrigin function"""
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))
