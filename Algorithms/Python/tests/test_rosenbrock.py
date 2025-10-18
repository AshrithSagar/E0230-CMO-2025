"""
tests/test_rosenbrock.py
"""

import numpy as np
from src.base import IterativeOptimiser
from src.functions import Rosenbrock
from src.optimisers import GradientDescent, NewtonMethod
from src.oracle import AbstractOracle, FirstOrderOracle, SecondOrderOracle
from src.stopping import GradientNormCriterion, MaxIterationsCriterion


def test_rosenbrock():
    dim: int = 2
    func = Rosenbrock(dim=dim, a=1.0, b=100.0)
    x0 = np.zeros(dim)
    criteria = [
        MaxIterationsCriterion(maxiter=int(1e6)),
        GradientNormCriterion(tol=1e-6),
    ]

    runs: list[tuple[IterativeOptimiser, AbstractOracle]] = [
        (GradientDescent(lr=1e-3), FirstOrderOracle(func)),
        (NewtonMethod(), SecondOrderOracle(func)),
    ]
    for optimiser, oracle in runs:
        info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria)
        assert np.allclose(info["x_star"], func.x_star, atol=1e-4), (
            f"Expected {func.x_star}, but got {info['x_star']}"
        )


if __name__ == "__main__":
    test_rosenbrock()
