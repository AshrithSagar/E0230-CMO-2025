"""
tests/test_rosenbrock.py
"""

import numpy as np
from src.functions import Rosenbrock
from src.optimisers import GradientDescent
from src.oracle import FirstOrderOracle


def test_rosenbrock():
    dim: int = 2
    func = Rosenbrock(dim=dim, a=1.0, b=100.0)
    oracle = FirstOrderOracle(func)
    optimizer = GradientDescent(lr=1e-3)
    x0 = np.zeros(dim)
    optimizer.run(oracle_fn=oracle, x0s=[x0], maxiter=int(1e6), tol=1e-6)
    assert np.allclose(optimizer.x_star, func.x_star, atol=1e-4), (
        f"Expected {func.x_star}, but got {optimizer.x_star}"
    )


if __name__ == "__main__":
    test_rosenbrock()
