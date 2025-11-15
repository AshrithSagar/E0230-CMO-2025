"""
tests/test_convex_quadratic.py
"""

import numpy as np
from cmo.functions import ConvexQuadratic
from cmo.helpers import optimise
from cmo.logging import Logger
from cmo.optimisers import ConjugateGradientMethod
from cmo.oracle import FirstOrderOracle
from cmo.stopping import GradientNormCriterion, MaxIterationsCriterion
from cmo.types import dtype

Logger.configure(level="DEBUG")


def test_convex_quadratic():
    dim: int = 2
    Q = np.array([[3, 0], [0, 1]], dtype=dtype)
    h = np.array([-3, -1], dtype=dtype)
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)
    oracle = FirstOrderOracle(func)
    optimiser = ConjugateGradientMethod()
    x0 = np.zeros(dim, dtype=dtype)
    criteria = [
        MaxIterationsCriterion(maxiter=int(1e2)),
        GradientNormCriterion(tol=1e-6),
    ]
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria, show_params=False)
    assert np.allclose(info.x_star, func.x_star, atol=1e-4), (
        f"Expected {func.x_star}, but got {info.x_star}"
    )


def test_convex_quadratic2():
    optimise(
        objective=ConvexQuadratic(
            dim=2,
            Q=np.array([[3, 0], [0, 1]], dtype=dtype),
            h=np.array([-3, -1], dtype=dtype),
        ),
        oracle=FirstOrderOracle,
        method=ConjugateGradientMethod(),
        x0=np.zeros(2, dtype=dtype),
        criteria=[
            MaxIterationsCriterion(maxiter=int(1e2)),
            GradientNormCriterion(tol=1e-6),
        ],
    )


if __name__ == "__main__":
    test_convex_quadratic()
    test_convex_quadratic2()
