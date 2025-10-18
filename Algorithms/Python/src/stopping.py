"""
src/stopping.py
=======
Stopping criteria
"""

from abc import ABC, abstractmethod
from typing import Generic, Iterable

import numpy as np

from .info import FirstOrderStepInfo, StepInfo, TStepInfo, ZeroOrderStepInfo


class StoppingCriterion(ABC, Generic[TStepInfo]):
    """An abstract base class to encapsulate various stopping criteria for iterative algorithms."""

    def reset(self):
        """Reset internal state, if any. Called at the beginning of each run."""
        pass

    @abstractmethod
    def check(self, info: TStepInfo) -> bool:
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

    def check(self, info: StepInfo) -> bool:
        return any(criterion.check(info) for criterion in self.criteria)


class MaxIterationsCriterion(StoppingCriterion):
    """
    Stops when the maximum number of iterations is reached.

    `k >= maxiter`
    """

    def __init__(self, maxiter: int = 1000):
        self.maxiter = int(maxiter)
        """Maximum number of iterations."""

    def check(self, info: StepInfo) -> bool:
        return bool(info.k >= self.maxiter)


class GradientNormCriterion(StoppingCriterion):
    """
    Stops when the gradient norm is below a specified tolerance.

    `||f'(x_k)|| < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the gradient norm."""

    def check(self, info: FirstOrderStepInfo) -> bool:
        return bool(np.linalg.norm(info.dfx) < self.tol)


class FunctionValueCriterion(StoppingCriterion):
    """
    Stops when the function value is below a specified tolerance.

    `f(x_k) < tol`
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = float(tol)
        """Tolerance for the function value."""

    def check(self, info: ZeroOrderStepInfo) -> bool:
        return bool(info.fx < self.tol)
