"""
Optimisation test problems
=======
src/cmo/functions.py

References
-------
- https://www.sfu.ca/~ssurjano/optimization.html
- https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from .types import floatMat, floatVec


class Function(ABC):
    """
    An abstract base class for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    def __init__(self, dim: int):
        self.dim: int = dim
        """Dimension of the input `x` for the function."""

    def _verify_input(self, x: ArrayLike) -> floatVec:
        """Verify that the input `x` is of the correct shape and type."""
        _x: floatVec = np.asarray(x, dtype=np.float64)
        if _x.shape != (self.dim,):
            raise ValueError(f"Input must be of shape ({self.dim},), got {_x.shape}.")
        return _x

    @property
    def x_star(self) -> floatVec:
        """The known minimiser of the function, if available."""
        raise NotImplementedError

    @property
    def f_star(self) -> float:
        """The known minimum function value, if available, or try computing using `x_star`."""
        return self.eval(self.x_star)

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


class LinearFunction(Function):
    """A linear function of the form `f(x) = c^T x`"""

    def __init__(self, dim: int, c: floatVec):
        self.c: floatVec = np.asarray(c, dtype=np.float64)
        assert self.c.shape == (dim,), "c must be of shape (dim,)."
        super().__init__(dim=dim)

    def eval(self, x: floatVec) -> float:
        return float(self.c.T @ x)

    def grad(self, x: floatVec) -> floatVec:
        return self.c

    def hess(self, x: floatVec) -> floatMat:
        return np.zeros((self.dim, self.dim))


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

    @property
    def x_star(self) -> floatVec:
        return np.asarray(np.linalg.solve(self.Q, -self.h), dtype=np.float64)

    def eval(self, x: floatVec) -> float:
        return float(0.5 * x.T @ self.Q @ x + self.h.T @ x)

    def grad(self, x: floatVec) -> floatVec:
        return self.Q @ x + self.h

    def hess(self, x: floatVec) -> floatMat:
        return self.Q


class Rosenbrock(Function):
    """The Rosenbrock function"""

    def __init__(self, dim: int, a: float = 1.0, b: float = 100.0):
        self.a = float(a)
        self.b = float(b)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> floatVec:
        if self.dim == 2:
            return np.array([self.a, self.a**2])
        elif self.a == 0.0 or self.a == 1.0:
            return np.full(self.dim, self.a)
        else:
            raise NotImplementedError

    @property
    def f_star(self) -> float:
        if self.dim == 2:
            return 0.0
        elif self.a == 0.0 or self.a == 1.0:
            return 0.0
        else:
            raise NotImplementedError

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
        self.A = float(A)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return self.A * len(x) + sum(x**2 - self.A * np.cos(2 * np.pi * x))

    def grad(self, x: floatVec) -> floatVec:
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)

    def hess(self, x: floatVec) -> floatMat:
        return np.diag(2 + 4 * np.pi**2 * self.A * np.cos(2 * np.pi * x))


class Sphere(Function):
    """The Sphere function"""

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return np.sum(x**2)

    def grad(self, x: floatVec) -> floatVec:
        return 2 * x

    def hess(self, x: floatVec) -> floatMat:
        return 2 * np.eye(self.dim)


class Ackley(Function):
    """The Ackley function"""

    def __init__(self, dim: int, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.sum(x**2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(self.c * x)) / self.dim)
        return term1 + term2 + self.a + np.exp(1)


class DropWave(Function):
    """The Drop-Wave function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return -1.0

    def eval(self, x: floatVec) -> float:
        r2 = sum(x**2)
        r = np.sqrt(r2)
        return -(1 + np.cos(12 * r)) / (0.5 * r2 + 2)


class Eggholder(Function):
    """The Eggholder function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.array([512.0, 404.2319])

    @property
    def f_star(self) -> float:
        return -959.6407

    def eval(self, x: floatVec) -> float:
        a = x[1] + 47
        term1 = -a * np.sin(np.sqrt(abs(x[0] / 2 + a)))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - a)))
        return term1 + term2


class Griewank(Function):
    """The Griewank function"""

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return sum_term - prod_term + 1


class Levy(Function):
    """The Levy function"""

    @property
    def x_star(self) -> floatVec:
        return np.ones(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        term2 = sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        return term1 + term2 + term3


class Levy13(Function):
    """The Levy 13 function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.ones(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        term1 = np.sin(3 * np.pi * x[0]) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
        return term1 + term2 + term3


class Schwefel(Function):
    """The Schwefel function"""

    @property
    def x_star(self) -> floatVec:
        return np.full(self.dim, 420.9687)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return 418.9829 * self.dim - sum(x * np.sin(np.sqrt(abs(x))))


class Booth(Function):
    """The Booth function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.array([1.0, 3.0])

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class Beale(Function):
    """The Beale function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.array([3.0, 0.5])

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3


class Matyas(Function):
    """The Matyas function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return 0.26 * sum(x**2) - 0.48 * x[0] * x[1]


class SumSquares(Function):
    """The Sum Squares function"""

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return sum((i + 1) * x[i] ** 2 for i in range(len(x)))


class Zakharov(Function):
    """The Zakharov function"""

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        sum1 = sum(x**2)
        sum2 = sum(0.5 * (i + 1) * x[i] for i in range(len(x)))
        return sum1 + sum2**2 + sum2**4


class ThreeHumpCamel(Function):
    """The Three-Hump Camel function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> floatVec:
        return np.zeros(self.dim)

    @property
    def f_star(self) -> float:
        return 0.0

    def eval(self, x: floatVec) -> float:
        return (
            2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2
        )
