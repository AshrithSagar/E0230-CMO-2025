"""
tests/test_rosenbrock.py
"""

import numpy as np
from cmo.core.base import IterativeOptimiser
from cmo.core.oracle import FirstOrderOracle, Oracle, SecondOrderOracle
from cmo.core.stopping import (
    GradientNormCriterion,
    MaxIterationsCriterion,
    StoppingCriterionType,
)
from cmo.functions import Rosenbrock
from cmo.optimisers import (
    BFGSUpdate,
    DFPUpdate,
    GradientDescent,
    NewtonMethod,
    SR1Update,
)
from cmo.types import Vector
from cmo.utils.logging import Logger

Logger.configure(level="INFO")


def test_rosenbrock() -> None:
    dim = int(2)
    func = Rosenbrock(dim=dim, a=1.0, b=100.0)
    x0 = Vector(np.zeros(dim))
    criteria: StoppingCriterionType = [  # type: ignore
        MaxIterationsCriterion(maxiter=int(1e6)),
        GradientNormCriterion(tol=1e-6),
    ]

    f_oracle = FirstOrderOracle(func)
    s_oracle = SecondOrderOracle(func)
    runs: list[tuple[IterativeOptimiser, Oracle]] = [  # type: ignore
        (GradientDescent(lr=1e-3), f_oracle),
        (NewtonMethod(), s_oracle),
        (SR1Update(), f_oracle),
        (DFPUpdate(), f_oracle),
        (BFGSUpdate(), f_oracle),
    ]
    for optimiser, oracle in runs:
        info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria)
        assert np.allclose(info.x_star, func.x_star, atol=1e-4), (
            f"Expected {func.x_star}, but got {info.x_star}"
        )


if __name__ == "__main__":
    test_rosenbrock()
