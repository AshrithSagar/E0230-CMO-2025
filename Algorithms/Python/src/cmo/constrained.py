"""
Constrained optimisation
=======
src/cmo/constrained.py
"""

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
from .types import Matrix, Scalar, Vector, asVector, dtype


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
    ) -> tuple[Vector, Vector]:
        """
        Solve the equality-constrained quadratic program using KKT conditions.

        Parameters
        ----------
        problem : EqualityConstrainedQuadraticProgram
            The equality-constrained quadratic program to solve.

        Returns
        -------
        x : Vector
            The optimal solution vector.
        mu : Vector
            The Lagrange multipliers associated with the equality constraints.
        """

        Q: Matrix = problem.objective.Q
        h: Vector = problem.objective.h
        A_eq: Matrix = problem.constraint.A
        b_eq: Vector = problem.constraint.b

        n: int = Q.shape[0]
        m: int = A_eq.shape[0]

        KKT: Matrix = np.block([[Q, -A_eq.T], [A_eq, np.zeros((m, m))]])
        rhs: Vector = np.hstack([-h, b_eq])

        try:
            sol: Vector = asVector(np.linalg.solve(KKT, rhs))
        except np.linalg.LinAlgError as e:
            raise e

        x: Vector = sol[:n]
        mu: Vector = sol[n:]
        return x, mu


class ActiveSetMethod(LineSearchOptimiser[ActiveSetStepInfo]):
    """Active set method for quadratic programs."""

    StepInfoClass = ActiveSetStepInfo

    def __init__(self, problem: InequalityConstrainedQuadraticProgram, **kwargs):
        self.problem = problem
        super().__init__(**kwargs)

    @property
    def stopping(self) -> list[StoppingCriterionType]:
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
                tol: Scalar = 1e-8
                if np.allclose(v, 0, atol=tol, rtol=0):
                    mu = info.mu
                    if mu is None:
                        return False
                    if mu.size != len(W):
                        return False
                    min_idx = int(np.argmin(mu))
                    mu_min = Scalar(mu[min_idx])
                    if mu_min >= -tol:
                        return True
                    info.relax = W[min_idx]
                return False

        return super().stopping + [ActiveSetStoppingCriterion()]

    def direction(self, info: ActiveSetStepInfo) -> Vector:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        Q: Matrix = info.oracle._oracle_f.Q
        h: Vector = info.dfx
        dim: int = Q.shape[0]
        objective = ConvexQuadratic(dim=dim, Q=Q, h=h)

        W = info.ensure(info.W, fallback=[])
        A: Matrix = self.problem.constraint.A
        if len(W) == 0:  # No active constraints
            A_eq = np.zeros((0, A.shape[1]), dtype=dtype)
            b_eq = np.zeros((0,), dtype=dtype)
        else:
            A_eq: Matrix = A[W, :]
            b_eq = np.zeros((len(W),), dtype=dtype)
        constraint = LinearEqualityConstraintSet(A_eq, b_eq)

        ceqp = EqualityConstrainedQuadraticProgram(
            objective, FirstOrderOracle, constraint
        )
        v, mu = EQPSolver().solve(ceqp)
        info.mu = mu

        if np.allclose(v, 0) and len(W) > 0 and len(mu) > 0:
            if np.any(mu < 0):
                # Index in `mu` with the smallest (most negative) multiplier
                idx = int(np.argmin(mu))
                info.relax = W[idx]
        return v

    def step_length(self, info: ActiveSetStepInfo) -> Scalar:
        v = info.ensure(info.direction, message="Direction has not been computed.")
        if np.allclose(v, 0):
            return 0

        x = info.x
        W = info.ensure(info.W, fallback=[])
        A = self.problem.constraint.A
        b = self.problem.constraint.b
        m: int = A.shape[0]

        inds: list[int] = np.setdiff1d(np.arange(m), W).tolist()
        blocking: int = -1
        alpha = 1.0
        for i in inds:
            a_i = A[i, :]
            denom = Scalar(a_i @ v)
            if denom < 0:
                alpha_i = Scalar(b[i] - a_i @ x) / denom
                if alpha_i < alpha:
                    alpha = alpha_i
                    blocking = i

        if not blocking == -1:
            info.blocking = blocking

        info.W = W
        return alpha

    def step(self, info: ActiveSetStepInfo) -> ActiveSetStepInfo:
        info_next = super().step(info)
        info.W = info.ensure(info.W, message="Active set W has not been initialised.")
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
            block = (
                info_next.blocking if info_next.blocking is not None else info.blocking
            )
            if block == relax:
                info_next.relax = None
            else:
                info_next.W.remove(relax)
                info_next.relax = relax

        return info_next
