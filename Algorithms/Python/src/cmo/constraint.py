"""
Constraint utils
=======
src/cmo/constraint.py

References
-------
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Sequence

import numpy as np

from .types import floatMat, floatOrVec, floatVec


class ConstraintType(Enum):
    """Enumeration of constraint types."""

    EQUALITY = auto()
    """Equality constraint; `h(x) = 0`."""

    LESS_THAN_OR_EQUAL_TO = auto()
    """Inequality constraint; `g(x) <= 0`."""

    GREATER_THAN_OR_EQUAL_TO = auto()
    """Inequality constraint; `g(x) >= 0`."""

    @property
    def op(self) -> Callable[[floatOrVec, floatOrVec], bool]:
        """Returns the operation associated with the constraint type."""
        match self:
            case ConstraintType.EQUALITY:
                return lambda x, y: np.allclose(x, y)
            case ConstraintType.LESS_THAN_OR_EQUAL_TO:
                return lambda x, y: bool(np.less_equal(x, y).all())
            case ConstraintType.GREATER_THAN_OR_EQUAL_TO:
                return lambda x, y: bool(np.greater_equal(x, y).all())
            case _:
                raise ValueError(f"Unknown constraint type: {self}")


class AbstractConstraint[T: ConstraintType](ABC):
    """A base class for constraints."""

    ctype: T
    """The type of the constraint."""

    @abstractmethod
    def residual(self, x: Any) -> Any:
        """Computes the residual of the constraint at point `x`."""
        raise NotImplementedError

    @abstractmethod
    def is_satisfied(self, x: Any) -> Any:
        """Checks if the constraint is satisfied at point `x`."""
        raise NotImplementedError

    def is_equality(self) -> bool:
        """Checks if the constraint is an equality constraint."""
        return self.ctype == ConstraintType.EQUALITY

    def is_inequality(self) -> bool:
        """Checks if the constraint is an inequality constraint."""
        return self.ctype in {
            ConstraintType.LESS_THAN_OR_EQUAL_TO,
            ConstraintType.GREATER_THAN_OR_EQUAL_TO,
        }


class SingleConstraint[T: ConstraintType](AbstractConstraint[T], ABC):
    """A class representing a single constraint."""

    @abstractmethod
    def residual(self, x: floatVec) -> float:
        raise NotImplementedError

    def is_satisfied(self, x: floatVec) -> bool:
        return self.ctype.op(self.residual(x), 0)

    def is_active(self, x: floatVec, tol: float = 1e-8) -> bool:
        """Checks if the constraint is active at point `x` within a tolerance."""
        if self.is_equality():
            return self.is_satisfied(x)
        else:
            return abs(self.residual(x)) <= tol


class MultiConstraint[T: ConstraintType](AbstractConstraint[T], ABC):
    """A class representing multiple (similar) constraints."""

    constraints: Sequence[SingleConstraint[T]]
    """The sequence of constraints."""

    @abstractmethod
    def residual(self, x: floatVec) -> floatVec:
        raise NotImplementedError

    def is_satisfied(self, x: floatVec) -> bool:
        residual = self.residual(x)
        return self.ctype.op(residual, np.zeros_like(residual))

    def active_set(
        self, x: floatVec, tol: float = 1e-8
    ) -> Sequence[SingleConstraint[T]]:
        """Returns the set of active constraints at point `x` within a tolerance."""
        return [c for c in self.constraints if c.is_active(x, tol)]


class LinearConstraint[T: ConstraintType](SingleConstraint[T]):
    """A single linear constraint with residual `(a^T x - b)`."""

    def __init__(self, a: floatVec, b: float):
        self.a = np.asarray(a, dtype=np.double)
        self.b = float(b)

    def residual(self, x: floatVec) -> float:
        return float(self.a @ x) - self.b


class LinearInequalityConstraint(
    LinearConstraint[ConstraintType.LESS_THAN_OR_EQUAL_TO]
):
    """A class representing a single linear constraint of the form `a^T x <= b`."""

    ctype = ConstraintType.LESS_THAN_OR_EQUAL_TO


class LinearEqualityConstraint(LinearConstraint[ConstraintType.EQUALITY]):
    """A class representing a single linear constraint of the form `a^T x = b`."""

    ctype = ConstraintType.EQUALITY


class LinearConstraintSet[T: ConstraintType](MultiConstraint[T]):
    """A collection of linear constraints of the form `Ax <= b` or `Ax = b`."""

    constraint: type[LinearConstraint[T]]

    def __init__(self, A: floatMat, b: floatVec):
        self.A: floatMat = np.atleast_2d(A)
        self.b: floatVec = np.atleast_1d(b)
        assert self.A.shape[0] == self.b.shape[0], (
            "Incompatible dimensions between A and b."
        )

        self.constraints = [
            self.constraint(a_i, b_i) for a_i, b_i in zip(self.A, self.b)
        ]

    def residual(self, x: floatVec) -> floatVec:
        return self.A @ x - self.b


class LinearInequalityConstraintSet(
    LinearConstraintSet[ConstraintType.LESS_THAN_OR_EQUAL_TO]
):
    """A class representing linear constraints of the form `Ax <= b`."""

    ctype = ConstraintType.LESS_THAN_OR_EQUAL_TO
    constraint = LinearInequalityConstraint


class LinearEqualityConstraintSet(LinearConstraintSet[ConstraintType.EQUALITY]):
    """A class representing linear constraints of the form `Ax = b`."""

    ctype = ConstraintType.EQUALITY
    constraint = LinearEqualityConstraint
