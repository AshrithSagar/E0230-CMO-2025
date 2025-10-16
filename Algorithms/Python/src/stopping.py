"""
src/stopping.py
=======
Stopping criteria
"""

from abc import ABC, abstractmethod

import numpy as np

from .types import floatVec


class StoppingCriterion(ABC):
    """An abstract base class to encapsulate various stopping criteria for iterative algorithms."""

    def reset(self):
        """Reset internal state, if any. Called at the beginning of each run."""
        pass

    @abstractmethod
    def check(self, x: floatVec, k: int, f: float, grad: floatVec) -> bool:
        """
        Return True if the stopping criterion is met.
        Parameters:
            x: Current point `x_k`
            k: Current iteration number
            f: Current function value `f(x_k)`
            grad: Current gradient `f'(x_k)`
        """
        raise NotImplementedError


class CompositeCriterion(StoppingCriterion):
    """
    Combines multiple stopping criteria. Stops when any one of the criteria is met.
    """

    def __init__(self, *criteria: StoppingCriterion):
        self.criteria = criteria
        """Tuple of stopping criteria."""

    def reset(self):
        for criterion in self.criteria:
            criterion.reset()

    def check(self, x: floatVec, k: int, f: float, grad: floatVec) -> bool:
        return any(criterion.check(x, k, f, grad) for criterion in self.criteria)


class MaxIterationsCriterion(StoppingCriterion):
    """
    Stops when the maximum number of iterations is reached.

    `k >= max_iter`
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = int(max_iter)
        """Maximum number of iterations."""

    def check(self, x: floatVec, k: int, f: float, grad: floatVec) -> bool:
        return bool(k >= self.max_iter)


class GradientNormCriterion(StoppingCriterion):
    """
    Stops when the gradient norm is below a specified tolerance.

    `||f'(x_k)|| < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the gradient norm."""

    def check(self, x: floatVec, k: int, f: float, grad: floatVec) -> bool:
        return bool(np.linalg.norm(grad) < self.tol)


class FunctionValueCriterion(StoppingCriterion):
    """
    Stops when the function value is below a specified tolerance.

    `f(x_k) < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the function value."""

    def check(self, x: floatVec, k: int, f: float, grad: floatVec) -> bool:
        return bool(f < self.tol)
