"""
src/oracle.py
=======
Oracle utils
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .functions import Function
from .types import floatVec


class AbstractOracle(ABC):
    """An abstract base class for oracles."""

    def __init__(self, func: Function):
        self._func = func
        """The oracle function `f(x)`."""

        self.dim: int = func.dim
        """The dimension of the input space."""

        self.call_count: int = 0
        """
        Tracks the number of times the oracle function has been called.
        This is useful for the 'analytical complexity' of the algorithms.
        """

    @abstractmethod
    def __call__(self, x: floatVec) -> Tuple:
        """Evaluates the oracle function at `x`."""
        raise NotImplementedError

    def reset(self):
        """Resets the internal call count."""
        self.call_count = 0
        return self


class ZeroOrderOracle(AbstractOracle):
    """
    `f = oracle(x)`
    """

    def __call__(self, x: floatVec) -> tuple[float]:
        fx = self._func(x)

        self.call_count += 1
        return ((fx),)


class FirstOrderOracle(AbstractOracle):
    """
    `f(x), f'(x) = oracle(x)`
    """

    def __init__(self, func: Function):
        if func._grad is None:
            raise ValueError("Function must have a gradient for FirstOrderOracle.")
        super().__init__(func)

    def __call__(self, x: floatVec) -> tuple[float, floatVec]:
        fx = self._func(x)
        dfx = self._func.gradient(x)

        self.call_count += 1
        return fx, dfx


class SecondOrderOracle(AbstractOracle):
    """
    `f(x), f'(x), f''(x) = oracle(x)`
    """

    def __init__(self, func: Function):
        if func._grad is None:
            raise ValueError("Function must have a gradient for SecondOrderOracle.")
        if func._hess is None:
            raise ValueError("Function must have a Hessian for SecondOrderOracle.")
        super().__init__(func)

    def __call__(self, x: floatVec) -> tuple[float, floatVec, np.ndarray]:
        fx = self._func(x)
        dfx = self._func.gradient(x)
        d2fx = self._func.hessian(x)

        self.call_count += 1
        return fx, dfx, d2fx
