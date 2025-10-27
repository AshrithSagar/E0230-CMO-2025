"""
Problem classes
=======
src/cmo/problems.py
"""

from typing import Sequence, Union

from .constraint import (
    AbstractConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from .functions import ConvexQuadratic, Function, LinearFunction
from .oracle import AbstractOracle
from .types import floatVec


class AbstractProblem:
    """A base class for optimisation problems."""

    def __init__(self, objective: Function, oracle: type[AbstractOracle]):
        self.objective: Function = objective
        self.oracle = oracle(objective)


class ConstrainedProblem(AbstractProblem):
    """A class representing constrained optimisation problems."""

    def __init__(
        self,
        objective: Function,
        oracle: type[AbstractOracle],
        constraints: Sequence[AbstractConstraint],
    ):
        super().__init__(objective, oracle)
        self.constraints: Sequence[AbstractConstraint] = constraints

    def is_feasible(self, x: floatVec) -> bool:
        """Checks if point `x` satisfies all constraints."""
        return all(constraint.is_satisfied(x) for constraint in self.constraints)


class LinearProgram(ConstrainedProblem):
    """A class representing linear programming problems."""

    def __init__(
        self,
        objective: LinearFunction,
        oracle: type[AbstractOracle],
        constraints: Sequence[
            Union[LinearEqualityConstraint, LinearInequalityConstraint]
        ],
    ):
        super().__init__(objective, oracle, constraints)


class QuadraticProgram(ConstrainedProblem):
    """A class representing quadratic programming problems."""

    def __init__(
        self,
        objective: ConvexQuadratic,
        oracle: type[AbstractOracle],
        constraints: Sequence[
            Union[LinearEqualityConstraint, LinearInequalityConstraint]
        ],
    ):
        super().__init__(objective, oracle, constraints)


class EqualityConstrainedQuadraticProgram(QuadraticProgram):
    """A class representing convex equality quadratic problems."""

    def __init__(
        self,
        objective: ConvexQuadratic,
        oracle: type[AbstractOracle],
        constraint: LinearEqualityConstraint,
    ):
        super().__init__(objective, oracle, [constraint])


class InequalityConstrainedQuadraticProgram(QuadraticProgram):
    """A class representing convex inequality quadratic problems."""

    def __init__(
        self,
        objective: ConvexQuadratic,
        oracle: type[AbstractOracle],
        constraint: LinearInequalityConstraint,
    ):
        super().__init__(objective, oracle, [constraint])
