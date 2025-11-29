"""
Base functions
=======
src/cmo/functions/base.py

Base class for real-valued scalar mathematical functions
"""

from abc import ABC, abstractmethod

import numpy as np

from cmo.types import Matrix, Scalar, Vector, dtype


class Function(ABC):
    """
    An abstract base class for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    def __init__(self, dim: int) -> None:
        self.dim: int = dim
        """Dimension of the input `x` for the function."""

    @property
    def x_star(self) -> Vector:
        """The known minimiser of the function, if available."""
        raise NotImplementedError

    @property
    def f_star(self) -> Scalar:
        """The known minimum function value, if available, or try computing using `x_star`."""
        return self.eval(self.x_star)

    @abstractmethod
    def eval(self, x: Vector) -> Scalar:
        """Computes the function value at `x`."""
        raise NotImplementedError

    def grad(self, x: Vector) -> Vector:
        """Computes the gradient of the function at `x`."""
        raise NotImplementedError

    def hess(self, x: Vector) -> Matrix:
        """Computes the Hessian of the function at `x`."""
        raise NotImplementedError


class LinearFunction(Function):
    """A linear function of the form `f(x) = c^T x`"""

    def __init__(self, dim: int, c: Vector) -> None:
        self.c = Vector(c)
        assert self.c.shape == (dim,), "c must be of shape (dim,)."
        super().__init__(dim=dim)

    def eval(self, x: Vector) -> Scalar:
        return Scalar(self.c.T @ x)

    def grad(self, x: Vector) -> Vector:
        return self.c

    def hess(self, x: Vector) -> Matrix:
        return Matrix(np.zeros((self.dim, self.dim), dtype=dtype))


class ConvexQuadratic(Function):
    """A convex quadratic function of the form `f(x) = 0.5 * x^T Q x + h^T x`"""

    def __init__(self, dim: int, Q: Matrix, h: Vector) -> None:
        self.Q = Matrix(Q)
        self.h = Vector(h)
        assert self.Q.shape == (dim, dim), "Q must be of shape (dim, dim)."
        assert self.h.shape == (dim,), "h must be of shape (dim,)."

        # Check for symmetric positive definite
        if not np.allclose(self.Q, self.Q.T) or np.any(np.linalg.eigvals(self.Q) <= 0):
            raise ValueError("Q must be a symmetric positive definite matrix.")

        super().__init__(dim=dim)

    @property
    def x_star(self) -> Vector:
        return Vector(np.linalg.solve(self.Q, -self.h))

    def eval(self, x: Vector) -> Scalar:
        return Scalar(0.5 * x.T @ self.Q @ x + self.h.T @ x)

    def grad(self, x: Vector) -> Vector:
        return Vector(self.Q @ x + self.h)

    def hess(self, x: Vector) -> Matrix:
        return self.Q
