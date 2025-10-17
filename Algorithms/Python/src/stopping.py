"""
src/stopping.py
=======
Stopping criteria
"""

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .oracle import AbstractOracle, FirstOrderOracle, ZeroOrderOracle
from .types import floatVec


class StoppingCriterion(ABC):
    """An abstract base class to encapsulate various stopping criteria for iterative algorithms."""

    def reset(self):
        """Reset internal state, if any. Called at the beginning of each run."""
        pass

    @abstractmethod
    def check(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> bool:
        """
        Return True if the stopping criterion is met.
        [Required]: This method should be implemented by subclasses to define the specific stopping condition.
        Parameters:
            x: Current value of `x`, i.e., `x_k`.
            k: Current iteration number.
            oracle_fn: The oracle function to query for `f(x)`.
        """
        raise NotImplementedError


class CompositeCriterion(StoppingCriterion):
    """
    Combines multiple stopping criteria. Stops when any one of the criteria is met.
    """

    def __init__(self, criteria: Iterable[StoppingCriterion]):
        self.criteria = criteria
        """Iterable of stopping criteria."""

    def reset(self):
        for criterion in self.criteria:
            criterion.reset()

    def check(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> bool:
        return any(criterion.check(x, k, oracle_fn) for criterion in self.criteria)


class MaxIterationsCriterion(StoppingCriterion):
    """
    Stops when the maximum number of iterations is reached.

    `k >= maxiter`
    """

    def __init__(self, maxiter: int = 1000):
        self.maxiter = int(maxiter)
        """Maximum number of iterations."""

    def check(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> bool:
        return bool(k >= self.maxiter)


class GradientNormCriterion(StoppingCriterion):
    """
    Stops when the gradient norm is below a specified tolerance.

    `||f'(x_k)|| < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the gradient norm."""

    def check(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> bool:
        assert isinstance(oracle_fn, FirstOrderOracle), (
            f"{self.__class__.__name__} requires a FirstOrderOracle."
        )
        _, grad = oracle_fn(x)
        return bool(np.linalg.norm(grad) < self.tol)


class FunctionValueCriterion(StoppingCriterion):
    """
    Stops when the function value is below a specified tolerance.

    `f(x_k) < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the function value."""

    def check(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> bool:
        assert isinstance(oracle_fn, ZeroOrderOracle), (
            f"{self.__class__.__name__} requires a ZeroOrderOracle."
        )
        (f,) = oracle_fn(x)
        return bool(f < self.tol)
