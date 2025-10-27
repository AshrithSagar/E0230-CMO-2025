"""
src/problems.py
=======
Problem classes
"""

from .constraint import (
    AbstractConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from .functions import ConvexQuadratic, Function
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
        constraints: list[AbstractConstraint],
    ):
        super().__init__(objective, oracle)
        self.constraints = constraints

    def is_feasible(self, x: floatVec) -> bool:
        """Checks if point `x` satisfies all constraints."""
        return all(constraint.is_satisfied(x) for constraint in self.constraints)


class CEQP(ConstrainedProblem):
    """A class representing convex equality quadratic problems."""

    def __init__(
        self,
        objective: ConvexQuadratic,
        oracle: type[AbstractOracle],
        constraint: LinearEqualityConstraint,
    ):
        super().__init__(objective, oracle, [constraint])


class CIQP(ConstrainedProblem):
    """A class representing convex inequality quadratic problems."""

    def __init__(
        self,
        objective: ConvexQuadratic,
        oracle: type[AbstractOracle],
        constraint: LinearInequalityConstraint,
    ):
        super().__init__(objective, oracle, [constraint])
