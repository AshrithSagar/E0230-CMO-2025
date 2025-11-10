"""
tests/test_active_set.py
"""

import numpy as np
from cmo.constrained import ActiveSetMethod
from cmo.constraint import LinearInequalityConstraintSet
from cmo.functions import ConvexQuadratic
from cmo.logging import Logger
from cmo.oracle import FirstOrderOracle
from cmo.problems import InequalityConstrainedQuadraticProgram
from cmo.stopping import MaxIterationsCriterion

Logger.configure(level="INFO")


def test_active_set():
    dim: int = 2
    Q = np.array([[3.0, 0.0], [0.0, 2.0]])
    h = np.array([-2.0, -5.0])
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)

    A_ineq = np.array([[-1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [1.0, -1.0]])
    b_ineq = np.array([2.0, 6.0, 6.0, 2.0])
    constraint = LinearInequalityConstraintSet(A=A_ineq, b=b_ineq)
    problem = InequalityConstrainedQuadraticProgram(
        objective=func, oracle=FirstOrderOracle, constraint=constraint
    )

    optimiser = ActiveSetMethod(problem)
    oracle = FirstOrderOracle(func)
    x0 = np.zeros(dim)
    criteria = MaxIterationsCriterion(maxiter=int(1e2))

    assert constraint.is_satisfied(x0), "Initial point is not feasible."
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria)
    print(info)


if __name__ == "__main__":
    test_active_set()
