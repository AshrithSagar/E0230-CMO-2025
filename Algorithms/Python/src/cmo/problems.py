"""
Problem classes
=======
src/cmo/problems.py
"""

from typing import Any, Optional

from .constraint import (
    AbstractConstraint,
    LinearConstraintSet,
    LinearEqualityConstraintSet,
    LinearInequalityConstraintSet,
)
from .core import IterativeOptimiser
from .functions import ConvexQuadratic, Function, LinearFunction
from .info import RunInfo, StepInfo
from .oracle import Oracle
from .stopping import StoppingCriterionType
from .types import Vector


class AbstractProblem[F: Function, O: Oracle]:
    """A base class for optimisation problems."""

    def __init__(self, objective: F, oracle: type[O]) -> None:
        self.objective = objective
        self.oracle = oracle(objective)


class UnconstrainedProblem[F: Function, O: Oracle](AbstractProblem[F, O]):
    """A class representing unconstrained optimisation problems."""

    def solve[T: StepInfo[Any]](
        self,
        method: IterativeOptimiser[O, T],
        x0: Vector,
        criteria: Optional[StoppingCriterionType[T]] = None,
        show_params: bool = True,
    ) -> RunInfo[O, T]:
        """Solve the unconstrained optimisation problem."""
        info = method.run(
            oracle_fn=self.oracle, x0=x0, criteria=criteria, show_params=show_params
        )
        return info


class ConstrainedProblem[F: Function, O: Oracle, C: AbstractConstraint[Any]](
    AbstractProblem[F, O]
):
    """A class representing constrained optimisation problems."""

    def __init__(self, objective: F, oracle: type[O], constraint: C) -> None:
        super().__init__(objective, oracle)
        self.constraint = constraint


class LinearProgram[O: Oracle, C: LinearConstraintSet[Any]](
    ConstrainedProblem[LinearFunction, O, C]
):
    """A class representing linear programming problems."""


class QuadraticProgram[O: Oracle, C: LinearConstraintSet[Any]](
    ConstrainedProblem[ConvexQuadratic, O, C]
):
    """A class representing convex quadratic programming problems."""


class EqualityConstrainedQuadraticProgram[O: Oracle](
    QuadraticProgram[O, LinearEqualityConstraintSet]
):
    """A class representing convex equality-constrained quadratic programming problems."""


class InequalityConstrainedQuadraticProgram[O: Oracle](
    QuadraticProgram[O, LinearInequalityConstraintSet]
):
    """A class representing convex inequality-constrained quadratic programming problems."""
