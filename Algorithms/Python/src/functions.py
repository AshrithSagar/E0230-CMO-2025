"""
src/functions.py
=======
Optimisation test problems

References
-------
- https://www.sfu.ca/~ssurjano/optimization.html
"""

from abc import ABC, abstractmethod

import numpy as np

from .types import floatMat, floatVec


class Function(ABC):
    """
    An abstract base class for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    def __init__(self, dim: int):
        self.dim = dim
        """Dimension of the input `x` for the function."""

    def _verify_input(self, x: floatVec):
        """Verify that the input `x` is of the correct shape and type."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"Input must be of shape ({self.dim},), got {x.shape}.")
        return x

    @abstractmethod
    def eval(self, x: floatVec) -> float:
        """Computes the function value at `x`."""
        raise NotImplementedError

    def grad(self, x: floatVec) -> floatVec:
        """Computes the gradient of the function at `x`."""
        raise NotImplementedError

    def hess(self, x: floatVec) -> floatMat:
        """Computes the Hessian of the function at `x`."""
        raise NotImplementedError


class ConvexQuadratic(Function):
    """A convex quadratic function of the form `f(x) = 0.5 * x^T Q x + h^T x`"""

    def __init__(self, dim: int, Q: floatMat, h: floatVec):
        self.Q: floatMat = np.asarray(Q, dtype=np.float64)
        self.h: floatVec = np.asarray(h, dtype=np.float64)
        assert self.Q.shape == (dim, dim), "Q must be of shape (dim, dim)."
        assert self.h.shape == (dim,), "h must be of shape (dim,)."

        # Check for symmetric positive definite
        if not np.allclose(self.Q, self.Q.T) or np.any(np.linalg.eigvals(self.Q) <= 0):
            raise ValueError("Q must be a symmetric positive definite matrix.")

        super().__init__(dim=dim)

    def eval(self, x: floatVec) -> float:
        return float(0.5 * x.T @ self.Q @ x + self.h.T @ x)

    def grad(self, x: floatVec) -> floatVec:
        return self.Q @ x + self.h

    def hess(self, x: floatVec) -> floatMat:
        return self.Q


class Rosenbrock(Function):
    """The Rosenbrock function"""

    def __init__(self, dim: int, a: float = 1.0, b: float = 100.0):
        self.a = a
        self.b = b
        super().__init__(dim=dim)

    def eval(self, x: floatVec) -> float:
        return sum((self.a - x[:-1]) ** 2.0 + self.b * (x[1:] - x[:-1] ** 2.0) ** 2.0)

    def grad(self, x: floatVec) -> floatVec:
        grad = np.zeros_like(x)
        grad[0] = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2)
        for i in range(1, len(x) - 1):
            grad[i] = (
                2 * (x[i] - self.a)
                - 4 * self.b * x[i] * (x[i + 1] - x[i] ** 2)
                + 2 * self.b * (x[i] - x[i - 1] ** 2)
            )
        grad[-1] = 2 * self.b * (x[-1] - x[-2] ** 2)
        return grad

    def hess(self, x: floatVec) -> floatMat:
        H = np.zeros((len(x), len(x)))
        H[0, 0] = 2 - 4 * self.b * (x[1] - 3 * x[0] ** 2)
        H[0, 1] = -4 * self.b * x[0]
        for i in range(1, len(x) - 1):
            H[i, i - 1] = -4 * self.b * x[i - 1]
            H[i, i] = 2 + 2 * self.b + 8 * self.b * x[i] ** 2
            H[i, i + 1] = -4 * self.b * x[i]
        H[-1, -2] = -4 * self.b * x[-2]
        H[-1, -1] = 2 * self.b
        return H


class Rastrigin(Function):
    """The Rastrigin function"""

    def __init__(self, dim: int, A: float = 10.0):
        self.A = A
        super().__init__(dim=dim)

    def eval(self, x: floatVec) -> float:
        return self.A * len(x) + sum(x**2 - self.A * np.cos(2 * np.pi * x))

    def grad(self, x: floatVec) -> floatVec:
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)

    def hess(self, x: floatVec) -> floatMat:
        return np.diag(2 + 4 * np.pi**2 * self.A * np.cos(2 * np.pi * x))
