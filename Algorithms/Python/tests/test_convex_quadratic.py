"""
tests/test_convex_quadratic.py
"""

import numpy as np
from src.functions import ConvexQuadratic
from src.optimisers import ConjugateGradientMethod
from src.oracle import FirstOrderOracle
from src.stopping import GradientNormCriterion, MaxIterationsCriterion


def test_convex_quadratic():
    dim: int = 2
    Q = np.array([[3, 0], [0, 1]], dtype=np.float64)
    h = np.array([-3, -1], dtype=np.float64)
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)
    oracle = FirstOrderOracle(func)
    optimiser = ConjugateGradientMethod()
    x0 = np.zeros(dim)
    criteria = [
        MaxIterationsCriterion(maxiter=int(1e2)),
        GradientNormCriterion(tol=1e-6),
    ]
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria, show_params=False)
    assert np.allclose(info["x_star"], func.x_star, atol=1e-4), (
        f"Expected {func.x_star}, but got {info['x_star']}"
    )


if __name__ == "__main__":
    test_convex_quadratic()
