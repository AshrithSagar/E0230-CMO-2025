"""
Oracle utils
=======
src/cmo/core/oracle.py
"""

from typing import Generic, Self, TypeVar

from cmo.functions.protocol import (
    FirstOrderFunctionProto,
    FirstOrderFunctionProtoT_co,
    FunctionProto,
    FunctionProtoT_co,
    SecondOrderFunctionProto,
    SecondOrderFunctionProtoT_co,
    ZeroOrderFunctionProto,
    ZeroOrderFunctionProtoT_co,
)
from cmo.types import Matrix, Scalar, Vector
from cmo.utils.logging import logger


class Oracle(Generic[FunctionProtoT_co]):
    """A base class for oracles."""

    def __init__(self, func: FunctionProtoT_co) -> None:
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


class ZeroOrderOracle(Oracle[ZeroOrderFunctionProtoT_co]):
    """`f = oracle(x)`"""

    def __init__(self, func: ZeroOrderFunctionProtoT_co) -> None:
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


class FirstOrderOracle(ZeroOrderOracle[FirstOrderFunctionProtoT_co]):
    """`f(x), f'(x) = oracle(x)`"""

    def __init__(self, func: FirstOrderFunctionProtoT_co) -> None:
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


class SecondOrderOracle(FirstOrderOracle[SecondOrderFunctionProtoT_co]):
    """`f(x), f'(x), f''(x) = oracle(x)`"""

    def __init__(self, func: SecondOrderFunctionProtoT_co) -> None:
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


OracleT_co = TypeVar(
    "OracleT_co",
    bound=Oracle[FunctionProto],
    covariant=True,
)
ZeroOrderOracleT_co = TypeVar(
    "ZeroOrderOracleT_co",
    bound=ZeroOrderOracle[ZeroOrderFunctionProto],
    covariant=True,
)
FirstOrderOracleT_co = TypeVar(
    "FirstOrderOracleT_co",
    bound=FirstOrderOracle[FirstOrderFunctionProto],
    covariant=True,
)
SecondOrderOracleT_co = TypeVar(
    "SecondOrderOracleT_co",
    bound=SecondOrderOracle[SecondOrderFunctionProto],
    covariant=True,
)
