"""
Problem classes
=======
src/cmo/problems.py
"""

from typing import Any, Optional

from cmo.constrained.constraint import (
    AbstractConstraint,
    LinearConstraintSet,
    LinearEqualityConstraintSet,
    LinearInequalityConstraintSet,
)
from cmo.core.base import IterativeOptimiser
from cmo.core.info import RunInfo, StepInfo
from cmo.core.oracle import Oracle
from cmo.core.stopping import StoppingCriterionType
from cmo.functions import ConvexQuadratic, LinearFunction
from cmo.functions.protocol import FunctionProto
from cmo.types import Vector


class AbstractProblem[F: FunctionProto, O: Oracle[Any]]:
    """A base class for optimisation problems."""

    def __init__(self, objective: F, oracle: type[O]) -> None:
        self.objective = objective
        self.oracle = oracle(objective)


class UnconstrainedProblem[F: FunctionProto, O: Oracle[Any]](AbstractProblem[F, O]):
    """A class representing unconstrained optimisation problems."""

    def solve[S: StepInfo[Any, Any]](
        self,
        method: IterativeOptimiser[F, O, S],
        x0: Vector,
        criteria: Optional[StoppingCriterionType[F, O, S]] = None,
        show_params: bool = True,
    ) -> RunInfo[F, O, S]:
        """Solve the unconstrained optimisation problem."""
        info = method.run(
            oracle_fn=self.oracle, x0=x0, criteria=criteria, show_params=show_params
        )
        return info


class ConstrainedProblem[F: FunctionProto, O: Oracle[Any], C: AbstractConstraint[Any]](
    AbstractProblem[F, O]
):
    """A class representing constrained optimisation problems."""

    def __init__(self, objective: F, oracle: type[O], constraint: C) -> None:
        super().__init__(objective, oracle)
        self.constraint = constraint


class LinearProgram[O: Oracle[Any], C: LinearConstraintSet[Any]](
    ConstrainedProblem[LinearFunction, O, C]
):
    """A class representing linear programming problems."""


class QuadraticProgram[O: Oracle[Any], C: LinearConstraintSet[Any]](
    ConstrainedProblem[ConvexQuadratic, O, C]
):
    """A class representing convex quadratic programming problems."""


class EqualityConstrainedQuadraticProgram[O: Oracle[Any]](
    QuadraticProgram[O, LinearEqualityConstraintSet]
):
    """A class representing convex equality-constrained quadratic programming problems."""


class InequalityConstrainedQuadraticProgram[O: Oracle[Any]](
    QuadraticProgram[O, LinearInequalityConstraintSet]
):
    """A class representing convex inequality-constrained quadratic programming problems."""
