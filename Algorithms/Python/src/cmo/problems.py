"""
Problem classes
=======
src/cmo/problems.py
"""

from typing import Optional

from .base import IterativeOptimiser
from .constraint import (
    AbstractConstraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from .functions import ConvexQuadratic, Function, LinearFunction
from .oracle import AbstractOracle
from .stopping import StoppingCriterionType
from .types import floatVec


class AbstractProblem[F: Function]:
    """A base class for optimisation problems."""

    def __init__(self, objective: F, oracle: type[AbstractOracle]):
        self.objective = objective
        self.oracle = oracle(objective)


class UnconstrainedProblem[F: Function, M: IterativeOptimiser](AbstractProblem[F]):
    """A class representing unconstrained optimisation problems."""

    def solve(
        self,
        method: M,
        x0: floatVec,
        criteria: Optional[StoppingCriterionType] = None,
        show_params: bool = True,
    ):
        """Solve the unconstrained optimisation problem."""
        info = method.run(
            oracle_fn=self.oracle, x0=x0, criteria=criteria, show_params=show_params
        )
        return info


class ConstrainedProblem[F: Function, C: AbstractConstraint](AbstractProblem[F]):
    """A class representing constrained optimisation problems."""

    def __init__(self, objective: F, oracle: type[AbstractOracle], constraint: C):
        super().__init__(objective, oracle)
        self.constraint = constraint


class LinearProgram[C: LinearConstraint](ConstrainedProblem[LinearFunction, C]):
    """A class representing linear programming problems."""


class QuadraticProgram[C: LinearConstraint](ConstrainedProblem[ConvexQuadratic, C]):
    """A class representing convex quadratic programming problems."""


class EqualityConstrainedQuadraticProgram(QuadraticProgram[LinearEqualityConstraint]):
    """A class representing convex equality-constrained quadratic programming problems."""


class InequalityConstrainedQuadraticProgram(
    QuadraticProgram[LinearInequalityConstraint]
):
    """A class representing convex inequality-constrained quadratic programming problems."""
