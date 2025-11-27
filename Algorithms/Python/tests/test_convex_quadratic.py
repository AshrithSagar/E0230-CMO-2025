"""
tests/test_convex_quadratic.py
"""

import numpy as np
from cmo.core.oracle import FirstOrderOracle
from cmo.core.stopping import (
    GradientNormCriterion,
    MaxIterationsCriterion,
    StoppingCriterionType,
)
from cmo.functions import ConvexQuadratic
from cmo.optimisers import ConjugateGradientMethod
from cmo.types import Matrix, Vector
from cmo.utils.helpers import optimise
from cmo.utils.logging import Logger

Logger.configure(level="DEBUG")


def test_convex_quadratic() -> None:
    dim = int(2)
    Q = Matrix([[3, 0], [0, 1]])
    h = Vector([-3, -1])
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)
    oracle = FirstOrderOracle(func)
    optimiser = ConjugateGradientMethod()
    x0 = Vector(np.zeros(dim))
    criteria: StoppingCriterionType = [  # type: ignore
        MaxIterationsCriterion(maxiter=int(1e2)),
        GradientNormCriterion(tol=1e-6),
    ]
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria, show_params=False)
    assert np.allclose(info.x_star, func.x_star, atol=1e-4), (
        f"Expected {func.x_star}, but got {info.x_star}"
    )


def test_convex_quadratic2() -> None:
    optimise(
        objective=ConvexQuadratic(
            dim=2,
            Q=Matrix([[3, 0], [0, 1]]),
            h=Vector([-3, -1]),
        ),
        oracle=FirstOrderOracle,
        method=ConjugateGradientMethod(),
        x0=Vector(np.zeros(2)),
        criteria=[
            MaxIterationsCriterion(maxiter=int(1e2)),
            GradientNormCriterion(tol=1e-6),
        ],
    )


if __name__ == "__main__":
    test_convex_quadratic()
    test_convex_quadratic2()
