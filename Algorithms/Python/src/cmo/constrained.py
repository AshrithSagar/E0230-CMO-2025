"""
Constrained optimisation
=======
src/cmo/constrained.py
"""

import numpy as np

from .problems import EqualityConstrainedQuadraticProgram
from .types import floatMat, floatVec


class EQPSolver:
    """
    Equality-constrained quadratic program solver.

    minimize `(1/2) x^T Q x + h^T x`\\
    subject to `A_eq x = b_eq`
    """

    def solve(
        self, problem: EqualityConstrainedQuadraticProgram
    ) -> tuple[floatVec, floatVec]:
        Q: floatMat = problem.objective.Q
        h: floatVec = problem.objective.h
        A_eq: floatMat = problem.constraint.A
        b_eq: floatVec = problem.constraint.b

        n: int = Q.shape[0]
        m: int = A_eq.shape[0]

        KKT: floatMat = np.block([[Q, A_eq.T], [A_eq, np.zeros((m, m))]])
        rhs: floatVec = -np.hstack([h, b_eq])
        sol: floatVec = np.asarray(np.linalg.solve(KKT, rhs), dtype=np.float64)

        x: floatVec = sol[:n]
        lam: floatVec = sol[n:]
        return x, lam
