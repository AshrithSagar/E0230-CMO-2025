"""
src/functions.py
=======
Optimisation test problems

References
-------
- https://www.sfu.ca/~ssurjano/optimization.html
"""

from functools import partial
from typing import Callable, Optional

import numpy as np

from .types import floatMat, floatVec


class Function:
    """
    A base class for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    def __init__(
        self,
        dim: int,
        func: Callable[[floatVec], float],
        grad: Optional[Callable[[floatVec], floatVec]] = None,
        hess: Optional[Callable[[floatVec], floatMat]] = None,
    ):
        self.dim = dim
        """Dimension of the input `x` for the function."""

        self._func = func
        """The function `f(x)`."""

        self._grad = grad
        """The gradient function `f'(x)`."""

        self._hess = hess
        """The Hessian function `f''(x)`."""

    def __call__(self, x: floatVec) -> float:
        """Computes the function value at `x`."""
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.dim,), f"x must be of shape ({self.dim},)"

        return self._func(x)

    def gradient(self, x: floatVec) -> floatVec:
        """Computes the gradient at `x`."""
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.dim,), f"x must be of shape ({self.dim},)"

        if self._grad is None:
            raise NotImplementedError(
                f"Gradient not implemented for {self.__class__.__name__}."
            )
        return self._grad(x)

    def hessian(self, x: floatVec) -> floatMat:
        """Computes the Hessian at `x`."""
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.dim,), f"x must be of shape ({self.dim},)"

        if self._hess is None:
            raise NotImplementedError(
                f"Hessian not implemented for {self.__class__.__name__}."
            )
        return self._hess(x)


class ConvexQuadratic(Function):
    """A convex quadratic function of the form `f(x) = 0.5 * x^T Q x + h^T x`"""

    def __init__(self, dim: int, Q: floatMat, h: floatVec):
        Q = np.asarray(Q, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        assert Q.shape == (dim, dim), "Q must be of shape (dim, dim)."
        assert h.shape == (dim,), "h must be of shape (dim,)."

        # Check for symmetric positive definite
        if not np.allclose(Q, Q.T) or np.any(np.linalg.eigvals(Q) <= 0):
            raise ValueError("Q must be a symmetric positive definite matrix.")

        def func(x: floatVec, Q: floatMat, h: floatVec) -> float:
            return float(0.5 * x.T @ Q @ x + h.T @ x)

        def grad(x: floatVec, Q: floatMat, h: floatVec) -> floatVec:
            return Q @ x + h

        def hess(x: floatVec, Q: floatMat, h: floatVec) -> floatMat:
            return Q

        super().__init__(
            dim=dim,
            func=partial(func, Q=Q, h=h),
            grad=partial(grad, Q=Q, h=h),
            hess=partial(hess, Q=Q, h=h),
        )


class Rosenbrock(Function):
    """The Rosenbrock function"""

    def __init__(self, dim: int, a: float = 1.0, b: float = 100.0):
        def func(x: floatVec, a: float, b: float) -> float:
            return sum((a - x[:-1]) ** 2.0 + b * (x[1:] - x[:-1] ** 2.0) ** 2.0)

        def grad(x: floatVec, a: float, b: float) -> floatVec:
            grad = np.zeros_like(x)
            grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
            for i in range(1, len(x) - 1):
                grad[i] = (
                    2 * (x[i] - a)
                    - 4 * b * x[i] * (x[i + 1] - x[i] ** 2)
                    + 2 * b * (x[i] - x[i - 1] ** 2)
                )
            grad[-1] = 2 * b * (x[-1] - x[-2] ** 2)
            return grad

        def hess(x: floatVec, a: float, b: float) -> floatMat:
            H = np.zeros((len(x), len(x)))
            H[0, 0] = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
            H[0, 1] = -4 * b * x[0]
            for i in range(1, len(x) - 1):
                H[i, i - 1] = -4 * b * x[i - 1]
                H[i, i] = 2 + 2 * b + 8 * b * x[i] ** 2
                H[i, i + 1] = -4 * b * x[i]
            H[-1, -2] = -4 * b * x[-2]
            H[-1, -1] = 2 * b
            return H

        super().__init__(
            dim=dim,
            func=partial(func, a=a, b=b),
            grad=partial(grad, a=a, b=b),
            hess=partial(hess, a=a, b=b),
        )


class Rastrigin(Function):
    """The Rastrigin function"""

    def __init__(self, dim: int, A: float = 10.0):
        def func(x: floatVec, A: float) -> float:
            return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

        def grad(x: floatVec, A: float) -> floatVec:
            return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

        def hess(x: floatVec, A: float) -> floatMat:
            return np.diag(2 + 4 * np.pi**2 * A * np.cos(2 * np.pi * x))

        super().__init__(
            dim=dim,
            func=partial(func, A=A),
            grad=partial(grad, A=A),
            hess=partial(hess, A=A),
        )
