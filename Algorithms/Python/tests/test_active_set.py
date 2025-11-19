"""
tests/test_active_set.py
"""

import numpy as np
from cmo.constrained import ActiveSetMethod
from cmo.constraint import LinearInequalityConstraintSet
from cmo.functions import ConvexQuadratic
from cmo.logging import Logger, console
from cmo.oracle import FirstOrderOracle
from cmo.problems import InequalityConstrainedQuadraticProgram
from cmo.stopping import MaxIterationsCriterion
from cmo.types import Matrix, Vector, dtype

Logger.configure(level="DEBUG")


def test_active_set():
    dim: int = 2
    Q = Matrix([[3.0, 0.0], [0.0, 2.0]], dtype=dtype)
    h = Vector([-2.0, -5.0], dtype=dtype)
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)

    A_ineq = Matrix([[-1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [1.0, -1.0]], dtype=dtype)
    b_ineq = Vector([2.0, 6.0, 6.0, 2.0], dtype=dtype)
    constraint = LinearInequalityConstraintSet(A=A_ineq, b=b_ineq)
    problem = InequalityConstrainedQuadraticProgram(
        objective=func, oracle=FirstOrderOracle, constraint=constraint
    )

    optimiser = ActiveSetMethod(problem)
    oracle = FirstOrderOracle(func)
    x0 = Vector(np.zeros(dim, dtype=dtype))
    criteria = MaxIterationsCriterion(maxiter=int(1e2))

    assert constraint.is_satisfied(x0), "Initial point is not feasible."
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria)
    console.print(str(info))


if __name__ == "__main__":
    test_active_set()
