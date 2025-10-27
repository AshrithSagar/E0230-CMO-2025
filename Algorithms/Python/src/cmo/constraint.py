"""
Constraint utils
=======
src/cmo/constraint.py
"""

from abc import ABC, abstractmethod

from .types import floatMat, floatVec


class AbstractConstraint(ABC):
    """An abstract base class for constraints."""

    @abstractmethod
    def is_satisfied(self, x: floatVec) -> bool:
        """Checks if the constraint is satisfied at point `x`."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class LinearInequalityConstraint(AbstractConstraint):
    """A class representing linear constraints of the form `Ax <= b`."""

    def __init__(self, A: floatMat, b: floatVec):
        self.A: floatMat = A
        self.b: floatVec = b

    def is_satisfied(self, x: floatVec) -> bool:
        return all(a_i @ x <= b_i for a_i, b_i in zip(self.A, self.b))


class LinearEqualityConstraint(AbstractConstraint):
    """A class representing linear constraints of the form `Ax = b`."""

    def __init__(self, A: floatMat, b: floatVec):
        self.A: floatMat = A
        self.b: floatVec = b

    def is_satisfied(self, x: floatVec) -> bool:
        return all(a_i @ x == b_i for a_i, b_i in zip(self.A, self.b))
