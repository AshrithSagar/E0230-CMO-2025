"""
Stopping criteria
=======
src/cmo/core/stopping.py
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Union

import numpy as np

from cmo.core.info import FirstOrderStepInfo, StepInfo, ZeroOrderStepInfo
from cmo.core.oracle import FirstOrderOracle, Oracle, ZeroOrderOracle
from cmo.functions.protocol import (
    FirstOrderFunctionProto,
    FunctionProto,
    ZeroOrderFunctionProto,
)
from cmo.types import Scalar
from cmo.utils.logging import logger

type StoppingCriterionType[F: FunctionProto, O: Oracle[Any], S: StepInfo[Any, Any]] = (
    Union[
        "StoppingCriterion[F, O, S]",
        "CompositeCriterion",
        Iterable["StoppingCriterion[F, O, S]"],
    ]
)
"""Generic type alias for stopping criteria."""


class StoppingCriterion[F: FunctionProto, O: Oracle[Any], S: StepInfo[Any, Any]](ABC):
    """An abstract base class to encapsulate various stopping criteria for iterative algorithms."""

    def reset(self) -> None:
        """Reset internal state, if any. Called at the beginning of each run."""
        name = self.__class__.__name__
        logger.debug(f"Stopping criterion [yellow]{name}[/] has been reset.")

    @abstractmethod
    def check(self, info: S) -> bool:
        """
        Return True if the stopping criterion is met.
        [Required]: This method should be implemented by subclasses to define the specific stopping condition.
        Parameters:
            x: Current value of `x`, i.e., `x_k`.
            k: Current iteration number.
            oracle_fn: The oracle function to query for `f(x)`.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        name = self.__class__.__name__
        params = self.__dict__ if self.__dict__ else ""
        return f"{name}({params})"


class CompositeCriterion[
    F: FunctionProto,
    O: Oracle[Any],
    S: StepInfo[Any, Any],
](StoppingCriterion[F, O, S]):
    """
    Combines multiple stopping criteria. Stops when any one of the criteria is met.
    """

    def __init__(self, criteria: Iterable[StoppingCriterion[F, O, S]]) -> None:
        self.criteria = criteria
        """Iterable of stopping criteria."""

    def reset(self) -> None:
        for criterion in self.criteria:
            criterion.reset()

    def check(self, info: S) -> bool:
        logger.debug(
            f"Checking stopping criteria for {info.__class__.__name__}(k={info.k})"
        )
        return any(criterion.check(info) for criterion in self.criteria)


class MaxIterationsCriterion[
    F: FunctionProto,
    O: Oracle[Any],
    S: StepInfo[Any, Any],
](StoppingCriterion[F, O, S]):
    """
    Stops when the maximum number of iterations is reached.

    `k >= maxiter`
    """

    def __init__(self, maxiter: int = 1000) -> None:
        self.maxiter = int(maxiter)
        """Maximum number of iterations."""

    def check(self, info: S) -> bool:
        return bool(info.k >= self.maxiter)


class GradientNormCriterion[
    F: FirstOrderFunctionProto,
    O: FirstOrderOracle[Any],
    S: FirstOrderStepInfo[Any, Any],
](StoppingCriterion[F, O, S]):
    """
    Stops when the gradient norm is below a specified tolerance.

    `||f'(x_k)|| < tol`
    """

    def __init__(self, tol: Scalar = 1e-6) -> None:
        self.tol = Scalar(tol)
        """Tolerance for the gradient norm."""

    def check(self, info: S) -> bool:
        return bool(np.linalg.norm(info.dfx) < self.tol)


class FunctionValueCriterion[
    F: ZeroOrderFunctionProto,
    O: ZeroOrderOracle[Any],
    S: ZeroOrderStepInfo[Any, Any],
](StoppingCriterion[F, O, S]):
    """
    Stops when the function value is below a specified tolerance.

    `f(x_k) < tol`
    """

    def __init__(self, tol: Scalar = 1e-6) -> None:
        self.tol = Scalar(tol)
        """Tolerance for the function value."""

    def check(self, info: S) -> bool:
        return bool(info.fx < self.tol)
