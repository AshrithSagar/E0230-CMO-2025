"""
Oracle utils
=======
src/cmo/oracle.py
"""

from typing import Self

from .functions import Function
from .types import floatVec


class AbstractOracle:
    """A base class for oracles."""

    def __init__(self, func: Function):
        self._oracle_f = func
        """The objective function `f(x)`."""

        self.dim: int = func.dim
        """The dimension of the input space."""

        self.call_count: int = 0
        """
        Tracks the number of times the oracle function has been called.
        This is useful for the 'analytical complexity' of the algorithms.
        """

    def reset(self) -> Self:
        """Resets the internal call counts."""
        self.call_count = 0
        return self


class ZeroOrderOracle(AbstractOracle):
    """`f = oracle(x)`"""

    def __init__(self, func: Function):
        super().__init__(func)

        self.eval_call_count: int = 0
        """Tracks the number of function evaluations."""

    def eval(self, x: floatVec) -> float:
        self.call_count += 1
        self.eval_call_count += 1
        return self._oracle_f.eval(x)

    def reset(self) -> Self:
        self.eval_call_count = 0
        return super().reset()


class FirstOrderOracle(ZeroOrderOracle):
    """`f(x), f'(x) = oracle(x)`"""

    def __init__(self, func: Function):
        super().__init__(func)
        self.grad_call_count: int = 0
        """Tracks the number of gradient evaluations."""

    def grad(self, x: floatVec) -> floatVec:
        self.call_count += 1
        self.grad_call_count += 1
        return self._oracle_f.grad(x)

    def reset(self) -> Self:
        self.grad_call_count = 0
        return super().reset()


class SecondOrderOracle(FirstOrderOracle):
    """`f(x), f'(x), f''(x) = oracle(x)`"""

    def __init__(self, func: Function):
        super().__init__(func)
        self.hess_call_count: int = 0
        """Tracks the number of Hessian evaluations."""

    def hess(self, x: floatVec) -> floatVec:
        self.call_count += 1
        self.hess_call_count += 1
        return self._oracle_f.hess(x)

    def reset(self) -> Self:
        self.hess_call_count = 0
        return super().reset()
