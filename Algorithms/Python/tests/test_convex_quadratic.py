"""
tests/test_convex_quadratic.py
"""

import numpy as np
from cmo.functions import ConvexQuadratic
from cmo.optimisers import ConjugateGradientMethod
from cmo.oracle import FirstOrderOracle
from cmo.stopping import GradientNormCriterion, MaxIterationsCriterion


def test_convex_quadratic():
    dim: int = 2
    Q = np.array([[3, 0], [0, 1]], dtype=np.double)
    h = np.array([-3, -1], dtype=np.double)
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)
    oracle = FirstOrderOracle(func)
    optimiser = ConjugateGradientMethod()
    x0 = np.zeros(dim)
    criteria = [
        MaxIterationsCriterion(maxiter=int(1e2)),
        GradientNormCriterion(tol=1e-6),
    ]
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria, show_params=False)
    assert np.allclose(info.x_star, func.x_star, atol=1e-4), (
        f"Expected {func.x_star}, but got {info.x_star}"
    )


if __name__ == "__main__":
    test_convex_quadratic()
