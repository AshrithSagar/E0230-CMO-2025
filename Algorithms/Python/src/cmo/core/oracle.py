"""
Oracle utils
=======
src/cmo/core/oracle.py
"""

from typing import Self

from cmo.functions.protocol import (
    FirstOrderFunctionProto,
    FunctionProto,
    SecondOrderFunctionProto,
    ZeroOrderFunctionProto,
)
from cmo.types import Matrix, Scalar, Vector
from cmo.utils.logging import logger


class Oracle[F: FunctionProto]:
    """A base class for oracles."""

    def __init__(self, func: F) -> None:
        self.func = func
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
        logger.debug(f"Oracle [yellow]{self.__class__.__name__}[/] has been reset.")
        return self


class ZeroOrderOracle[F: ZeroOrderFunctionProto](Oracle[F]):
    """`f = oracle(x)`"""

    def __init__(self, func: F) -> None:
        super().__init__(func)

        self.eval_call_count: int = 0
        """Tracks the number of function evaluations."""

    def eval(self, x: Vector) -> Scalar:
        self.call_count += 1
        self.eval_call_count += 1
        return self.func.eval(x)

    def reset(self) -> Self:
        self.eval_call_count = 0
        return super().reset()


class FirstOrderOracle[F: FirstOrderFunctionProto](ZeroOrderOracle[F]):
    """`f(x), f'(x) = oracle(x)`"""

    def __init__(self, func: F) -> None:
        super().__init__(func)
        self.grad_call_count: int = 0
        """Tracks the number of gradient evaluations."""

    def grad(self, x: Vector) -> Vector:
        self.call_count += 1
        self.grad_call_count += 1
        return self.func.grad(x)

    def reset(self) -> Self:
        self.grad_call_count = 0
        return super().reset()


class SecondOrderOracle[F: SecondOrderFunctionProto](FirstOrderOracle[F]):
    """`f(x), f'(x), f''(x) = oracle(x)`"""

    def __init__(self, func: F) -> None:
        super().__init__(func)
        self.hess_call_count: int = 0
        """Tracks the number of Hessian evaluations."""

    def hess(self, x: Vector) -> Matrix:
        self.call_count += 1
        self.hess_call_count += 1
        return self.func.hess(x)

    def reset(self) -> Self:
        self.hess_call_count = 0
        return super().reset()
