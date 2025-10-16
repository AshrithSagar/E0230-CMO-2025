"""
tests/test_rosenbrock.py
"""

import numpy as np
from src.functions import Rosenbrock
from src.optimisers import SteepestGradientDescent
from src.oracle import FirstOrderOracle


def test_rosenbrock():
    dim: int = 2
    oracle = FirstOrderOracle(Rosenbrock(dim=dim))
    optimizer = SteepestGradientDescent()
    x0 = np.zeros(dim)
    optimizer.run(oracle_fn=oracle, x0s=[x0], maxiter=int(1e6), show_params=False)
    assert np.allclose(optimizer.x_star, np.ones(dim), atol=1e-4), (
        f"Expected {np.ones(dim)}, but got {optimizer.x_star}"
    )


if __name__ == "__main__":
    test_rosenbrock()
