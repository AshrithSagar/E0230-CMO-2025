"""
Function protocols
=======
src/cmo/functions/protocol.py

Protocols for real-valued scalar mathematical functions
"""

from typing import Protocol

from cmo.types import Matrix, Scalar, Vector


class FunctionProto(Protocol):
    """
    Protocol for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    dim: int
    """Dimension of the input `x` for the function."""


class ZeroOrderFunctionProto(FunctionProto, Protocol):
    def eval(self, x: Vector) -> Scalar:
        """The function value at `x`."""
        raise NotImplementedError


class FirstOrderFunctionProto(ZeroOrderFunctionProto, Protocol):
    def grad(self, x: Vector) -> Vector:
        """The gradient of the function at `x`."""
        raise NotImplementedError


class SecondOrderFunctionProto(FirstOrderFunctionProto, Protocol):
    def hess(self, x: Vector) -> Matrix:
        """The Hessian of the function at `x`."""
        raise NotImplementedError


class SupportsXStarProto(Protocol):
    """Protocol for functions that have a known minimiser `x_star`."""

    @property
    def x_star(self) -> Vector:
        """The known minimiser of the function."""
        raise NotImplementedError


class SupportsFStarProto(Protocol):
    """Protocol for functions that have a known minimum function value `f_star`."""

    @property
    def f_star(self) -> Scalar:
        """The known minimum function value."""
        raise NotImplementedError
