"""
Constrained optimisation
=======
src/cmo/constrained.py
"""

from typing import Optional

import numpy as np

from .base import IterativeOptimiser, LineSearchOptimiser
from .constraint import LinearEqualityConstraintSet
from .functions import ConvexQuadratic
from .info import ActiveSetStepInfo
from .oracle import FirstOrderOracle
from .problems import (
    EqualityConstrainedQuadraticProgram,
    InequalityConstrainedQuadraticProgram,
)
from .stopping import StoppingCriterion, StoppingCriterionType
from .types import floatMat, floatVec


class ConstrainedOptimiser(IterativeOptimiser):
    """Base class for iterative constrained optimisers."""


class EQPSolver:
    """
    Equality-constrained quadratic program solver.

    minimize `(1/2) x^T Q x + h^T x`\\
    subject to `A_eq x = b_eq`
    """

    def solve(
        self, problem: EqualityConstrainedQuadraticProgram
    ) -> tuple[floatVec, floatVec]:
        """
        Solve the equality-constrained quadratic program using KKT conditions.

        Parameters
        ----------
        problem : EqualityConstrainedQuadraticProgram
            The equality-constrained quadratic program to solve.

        Returns
        -------
        x : floatVec
            The optimal solution vector.
        mu : floatVec
            The Lagrange multipliers associated with the equality constraints.
        """

        Q: floatMat = problem.objective.Q
        h: floatVec = problem.objective.h
        A_eq: floatMat = problem.constraint.A
        b_eq: floatVec = problem.constraint.b

        n: int = Q.shape[0]
        m: int = A_eq.shape[0]

        KKT: floatMat = np.block([[Q, A_eq.T], [A_eq, np.zeros((m, m))]])
        rhs: floatVec = np.hstack([-h, b_eq])

        try:
            sol: floatVec = np.asarray(np.linalg.solve(KKT, rhs), dtype=np.double)
        except np.linalg.LinAlgError as e:
            raise e

        x: floatVec = sol[:n]
        mu: floatVec = sol[n:]
        return x, mu


class ActiveSetMethod(LineSearchOptimiser[ActiveSetStepInfo]):
    """Active set method for quadratic programs."""

    StepInfoClass = ActiveSetStepInfo

    def __init__(self, problem: InequalityConstrainedQuadraticProgram, **kwargs):
        self.problem = problem
        super().__init__(**kwargs)

    @property
    def stopping(self) -> Optional[StoppingCriterionType]:
        class ActiveSetStoppingCriterion(StoppingCriterion[ActiveSetStepInfo]):
            """
            Stops when all the Lagrange multipliers associated with the active constraints are non-negative.

            `mu_i >= 0 for all i in W_k`
            """

            def check(self, info: ActiveSetStepInfo) -> bool:
                v = info.direction
                if v is None:
                    return False
                W = info.W
                if W is None:
                    return False
                if np.allclose(v, 0):
                    mu = info.mu
                    if mu is None:
                        return False
                    mu_min: float = np.inf
                    relax: int = -1
                    for mu_i, W_i in zip(mu, W):
                        if mu_i < 0:
                            mu_min = min(mu_min, mu_i)
                            relax = W_i
                    if mu_min >= 0:
                        return True
                    info.relax = relax
                return False

        return ActiveSetStoppingCriterion()

    def direction(self, info: ActiveSetStepInfo) -> floatVec:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        Q: floatMat = info.oracle._oracle_f.Q
        h: floatVec = info.dfx
        dim: int = Q.shape[0]
        objective = ConvexQuadratic(dim=dim, Q=Q, h=h)

        W = info.W if info.W is not None else []
        A: floatMat = self.problem.constraint.A
        if len(W) == 0:  # No active constraints
            A_eq = np.zeros((0, A.shape[1]))
            b_eq = np.zeros((0,))
        else:
            A_eq = A[W, :]
            b_eq = np.zeros((len(W),))
        constraint = LinearEqualityConstraintSet(A_eq, b_eq)

        ceqp = EqualityConstrainedQuadraticProgram(
            objective, FirstOrderOracle, constraint
        )
        v, mu = EQPSolver().solve(ceqp)
        info.mu = mu
        return v

    def step_length(self, info: ActiveSetStepInfo) -> float:
        v = info.direction
        if v is None:
            raise ValueError("Direction has not been computed.")
        if np.allclose(v, 0):
            return 0

        x = info.x
        W = info.W if info.W is not None else []
        A = self.problem.constraint.A
        b = self.problem.constraint.b
        m: int = A.shape[0]

        inds: list[int] = np.setdiff1d(np.arange(m), W).tolist()
        blocking: int = -1
        alpha = 1.0
        for i in inds:
            a_i = A[i, :]
            denom = float(a_i @ v)
            if denom < 0:
                alpha_i = float(b[i] - a_i @ x) / denom
                if alpha_i < alpha:
                    alpha = alpha_i
                    blocking = i

        if not blocking == -1:
            info.blocking = blocking

        info.W = W
        return alpha

    def step(self, info: ActiveSetStepInfo) -> ActiveSetStepInfo:
        info_next = super().step(info)
        if info.W is None:
            raise ValueError("Active set W has not been initialised.")
        info_next.W = info.W.copy()

        blocking = info_next.blocking
        if blocking is None:
            blocking = info.blocking
        if blocking is not None and blocking not in info_next.W:
            info_next.W.append(blocking)
            info_next.blocking = blocking

        relax = info_next.relax
        if relax is None:
            relax = info.relax
        if relax is not None and relax in info_next.W:
            info_next.W.remove(relax)
            info_next.relax = relax

        return info_next
