"""
Constraint utils
=======
src/cmo/constraint.py

References
-------
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
"""

import operator
from abc import ABC, abstractmethod
from typing import Sequence

from .types import floatMat, floatVec


class AbstractConstraint(ABC):
    """An abstract base class for constraints."""

    @abstractmethod
    def is_satisfied(self, x: floatVec) -> bool:
        """Checks if the constraint is satisfied at point `x`."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class CompositeConstraint(AbstractConstraint):
    """A class representing a composite constraint made up of multiple constraints."""

    def __init__(self, constraints: Sequence[AbstractConstraint]):
        self.constraints: Sequence[AbstractConstraint] = constraints

    def is_satisfied(self, x: floatVec) -> bool:
        return all(constraint.is_satisfied(x) for constraint in self.constraints)


class BoundConstraint(AbstractConstraint):
    """A class representing bound constraints of the form `l <= x <= u`."""

    def __init__(self, lower: floatVec, upper: floatVec):
        self.lower: floatVec = lower
        self.upper: floatVec = upper

    def is_satisfied(self, x: floatVec) -> bool:
        return all(
            l_i <= x_i <= u_i for l_i, x_i, u_i in zip(self.lower, x, self.upper)
        )


class LinearConstraint(AbstractConstraint):
    """A class representing linear constraints."""

    def __init__(self, A: floatMat, b: floatVec, equality: bool = False):
        self.A: floatMat = A
        self.b: floatVec = b

        self._op = operator.eq if equality else operator.le

    def is_satisfied(self, x: floatVec) -> bool:
        return all(self._op(a_i @ x, b_i) for a_i, b_i in zip(self.A, self.b))

    def residual(self, x: floatVec) -> floatVec:
        """Computes the residuals of the constraints at point `x`."""
        return self.A @ x - self.b


class LinearInequalityConstraint(LinearConstraint):
    """A class representing linear constraints of the form `Ax <= b`."""

    def __init__(self, A: floatMat, b: floatVec):
        super().__init__(A, b, equality=False)


class LinearEqualityConstraint(LinearConstraint):
    """A class representing linear constraints of the form `Ax = b`."""

    def __init__(self, A: floatMat, b: floatVec):
        super().__init__(A, b, equality=True)
