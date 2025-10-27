"""
Problem classes
=======
src/cmo/problems.py
"""

from .constraint import (
    AbstractConstraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from .functions import ConvexQuadratic, Function, LinearFunction
from .oracle import AbstractOracle


class AbstractProblem[F: Function]:
    """A base class for optimisation problems."""

    def __init__(self, objective: F, oracle: type[AbstractOracle]):
        self.objective = objective
        self.oracle = oracle(objective)


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
