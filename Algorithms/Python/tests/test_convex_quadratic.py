"""
tests/test_convex_quadratic.py
"""

import numpy as np
from src.functions import ConvexQuadratic
from src.optimisers import ConjugateGradientMethod
from src.oracle import FirstOrderOracle


def test_convex_quadratic():
    dim: int = 2
    Q = np.array([[3, 0], [0, 1]])
    h = np.array([-3, -1])
    x_star = np.linalg.solve(Q, -h)
    oracle = FirstOrderOracle(ConvexQuadratic(dim=dim, Q=Q, h=h))
    optimizer = ConjugateGradientMethod()
    x0 = np.zeros(dim)
    optimizer.run(oracle_fn=oracle, x0s=[x0], maxiter=int(1e2), show_params=False)
    assert np.allclose(optimizer.x_star, x_star, atol=1e-4), (
        f"Expected {x_star}, but got {optimizer.x_star}"
    )


if __name__ == "__main__":
    test_convex_quadratic()
