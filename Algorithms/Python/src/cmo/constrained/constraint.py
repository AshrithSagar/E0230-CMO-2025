"""
Constraint utils
=======
src/cmo/constrained/constraint.py

References
-------
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from cmo.types import Matrix, Scalar, Vector


class ConstraintType(Enum):
    """Enumeration of constraint types."""

    EQUALITY = auto()
    """Equality constraint; `h(x) = 0`."""

    LESS_THAN_OR_EQUAL_TO = auto()
    """Inequality constraint; `g(x) <= 0`."""

    GREATER_THAN_OR_EQUAL_TO = auto()
    """Inequality constraint; `g(x) >= 0`."""

    @property
    def op(self) -> Callable[[Scalar | Vector, Scalar | Vector], bool]:
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


class AbstractConstraint[C: ConstraintType](ABC):
    """A base class for constraints."""

    ctype: C
    """The type of the constraint."""

    @abstractmethod
    def residual(self, x: Any) -> Any:
        """Computes the residual of the constraint at point `x`."""
        raise NotImplementedError

    @abstractmethod
    def project(self, x: Any) -> Any:
        """Projects point `x` onto the feasible region of the constraint."""
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


class SingleConstraint[C: ConstraintType](AbstractConstraint[C], ABC):
    """A class representing a single constraint."""

    @abstractmethod
    def residual(self, x: Vector) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def project(self, x: Vector) -> Vector:
        raise NotImplementedError

    def is_satisfied(self, x: Vector) -> bool:
        return self.ctype.op(self.residual(x), 0)

    def is_active(self, x: Vector, tol: Scalar = 1e-8) -> bool:
        """Checks if the constraint is active at point `x` within a tolerance."""
        if self.is_equality():
            return self.is_satisfied(x)
        else:
            return abs(self.residual(x)) <= tol


class MultiConstraint[C: ConstraintType](AbstractConstraint[C], ABC):
    """A class representing multiple (similar) constraints."""

    constraints: Sequence[SingleConstraint[C]]
    """The sequence of constraints."""

    @abstractmethod
    def residual(self, x: Vector) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def project(self, x: Vector) -> Vector:
        raise NotImplementedError

    def is_satisfied(self, x: Vector) -> bool:
        residual = self.residual(x)
        return self.ctype.op(residual, np.zeros_like(residual))

    def active_set(
        self, x: Vector, tol: Scalar = 1e-8
    ) -> Sequence[SingleConstraint[C]]:
        """Returns the set of active constraints at point `x` within a tolerance."""
        return [c for c in self.constraints if c.is_active(x, tol)]


class LowerBoundConstraint(
    AbstractConstraint[Literal[ConstraintType.GREATER_THAN_OR_EQUAL_TO]]
):
    """A class representing a single lower bound constraint of the form `x >= lb`."""

    ctype = ConstraintType.GREATER_THAN_OR_EQUAL_TO

    def __init__(self, lb: Vector) -> None:
        self.lb = Vector(lb)

    def residual(self, x: Vector) -> Vector:
        return Vector(x - self.lb)

    def project(self, x: Vector) -> Vector:
        return Vector(np.maximum(x, self.lb))

    def is_satisfied(self, x: Vector) -> bool:
        return bool(np.all(x >= self.lb))


class UpperBoundConstraint(
    AbstractConstraint[Literal[ConstraintType.LESS_THAN_OR_EQUAL_TO]]
):
    """A class representing a single upper bound constraint of the form `x <= ub`."""

    ctype = ConstraintType.LESS_THAN_OR_EQUAL_TO

    def __init__(self, ub: Vector) -> None:
        self.ub = Vector(ub)

    def residual(self, x: Vector) -> Vector:
        return Vector(x - self.ub)

    def project(self, x: Vector) -> Vector:
        return Vector(np.minimum(x, self.ub))

    def is_satisfied(self, x: Vector) -> bool:
        return bool(np.all(x <= self.ub))


class LinearConstraint[C: ConstraintType](SingleConstraint[C]):
    """A single linear constraint with residual `(a^T x - b)`."""

    def __init__(self, a: Vector, b: Scalar) -> None:
        self.a = Vector(a)
        self.b = Scalar(b)

    def residual(self, x: Vector) -> Scalar:
        return Scalar(self.a @ x) - self.b

    def _project(self, x: Vector) -> Vector:
        """Projects point `x` onto the hyperplane defined by the constraint."""
        a = self.a
        residual = self.residual(x)
        return Vector(x - (residual / np.dot(a, a)) * a)

    def project(self, x: Vector) -> Vector:
        return x if self.is_satisfied(x) else self._project(x)


class LinearInequalityConstraint(
    LinearConstraint[Literal[ConstraintType.LESS_THAN_OR_EQUAL_TO]]
):
    """A class representing a single linear constraint of the form `a^T x <= b`."""

    ctype = ConstraintType.LESS_THAN_OR_EQUAL_TO


class LinearEqualityConstraint(LinearConstraint[Literal[ConstraintType.EQUALITY]]):
    """A class representing a single linear constraint of the form `a^T x = b`."""

    ctype = ConstraintType.EQUALITY

    def project(self, x: Vector) -> Vector:
        return self._project(x)


class LinearConstraintSet[C: ConstraintType](MultiConstraint[C]):
    """A collection of linear constraints of the form `Ax <= b` or `Ax = b`."""

    constraint: type[LinearConstraint[C]]

    def __init__(self, A: Matrix, b: Vector) -> None:
        self.A = Matrix(A)
        self.b = Vector(b)
        assert self.A.shape[0] == self.b.shape[0], (
            "Incompatible dimensions between A and b."
        )

        self.constraints = [
            self.constraint(Vector(a_i), Scalar(b_i))
            for a_i, b_i in zip(self.A, self.b)
        ]

    def residual(self, x: Vector) -> Vector:
        return Vector(self.A @ x - self.b)

    def project(self, x: Vector) -> Vector:
        x_proj = x.copy()
        for constraint in self.constraints:
            x_proj = constraint.project(x_proj)
        return x_proj


class LinearInequalityConstraintSet(
    LinearConstraintSet[Literal[ConstraintType.LESS_THAN_OR_EQUAL_TO]]
):
    """A class representing linear constraints of the form `Ax <= b`."""

    ctype = ConstraintType.LESS_THAN_OR_EQUAL_TO
    constraint = LinearInequalityConstraint


class LinearEqualityConstraintSet(
    LinearConstraintSet[Literal[ConstraintType.EQUALITY]]
):
    """A class representing linear constraints of the form `Ax = b`."""

    ctype = ConstraintType.EQUALITY
    constraint = LinearEqualityConstraint

    def __init__(self, A: Matrix, b: Vector) -> None:
        super().__init__(A, b)
        self._AT_AAT_pinv: Optional[Matrix] = None

    def project(self, x: Vector) -> Vector:
        residual = self.residual(x)
        x_proj = Vector(x - self.AT_AAT_pinv @ residual)
        return x_proj

    @property
    def AT_AAT_pinv(self) -> Matrix:
        """Returns the pseudo-inverse of `A A^T`."""
        if self._AT_AAT_pinv is not None:
            return self._AT_AAT_pinv
        A = self.A
        AT = A.T
        _AT_AAT_pinv = AT @ np.linalg.pinv(A @ AT)
        self._AT_AAT_pinv = Matrix(_AT_AAT_pinv)
        return self._AT_AAT_pinv
