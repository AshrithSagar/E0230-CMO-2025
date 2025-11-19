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

from .types import Matrix, Scalar, Vector, dtype


class Function(ABC):
    """
    An abstract base class for real-valued scalar mathematical functions.

    `f: R^d -> R`
    """

    def __init__(self, dim: int):
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

    def __init__(self, dim: int, c: Vector):
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

    def __init__(self, dim: int, Q: Matrix, h: Vector):
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


class Rosenbrock(Function):
    """The Rosenbrock function"""

    def __init__(self, dim: int, a: Scalar = 1.0, b: Scalar = 100.0):
        self.a = Scalar(a)
        self.b = Scalar(b)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> Vector:
        if self.dim == 2:
            return Vector([self.a, self.a**2])
        elif self.a == 0.0 or self.a == 1.0:
            return Vector(np.full(self.dim, self.a, dtype=dtype))
        else:
            raise NotImplementedError

    @property
    def f_star(self) -> Scalar:
        if self.dim == 2:
            return 0.0
        elif self.a == 0.0 or self.a == 1.0:
            return 0.0
        else:
            raise NotImplementedError

    def eval(self, x: Vector) -> Scalar:
        return sum((self.a - x[:-1]) ** 2.0 + self.b * (x[1:] - x[:-1] ** 2.0) ** 2.0)

    def grad(self, x: Vector) -> Vector:
        grad = Vector(np.zeros_like(x, dtype=dtype))
        grad[0] = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2)
        for i in range(1, len(x) - 1):
            grad[i] = (
                2 * (x[i] - self.a)
                - 4 * self.b * x[i] * (x[i + 1] - x[i] ** 2)
                + 2 * self.b * (x[i] - x[i - 1] ** 2)
            )
        grad[-1] = 2 * self.b * (x[-1] - x[-2] ** 2)
        return grad

    def hess(self, x: Vector) -> Matrix:
        H = Matrix(np.zeros((len(x), len(x)), dtype=dtype))
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

    def __init__(self, dim: int, A: Scalar = 10.0):
        self.A = Scalar(A)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return self.A * len(x) + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))

    def grad(self, x: Vector) -> Vector:
        return Vector(2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x))

    def hess(self, x: Vector) -> Matrix:
        return Matrix(np.diag(2 + 4 * np.pi**2 * self.A * np.cos(2 * np.pi * x)))


class Sphere(Function):
    """The Sphere function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return Scalar(np.sum(x**2))

    def grad(self, x: Vector) -> Vector:
        return Vector(2 * x)

    def hess(self, x: Vector) -> Matrix:
        return Matrix(2 * np.eye(self.dim, dtype=dtype))


class Ackley(Function):
    """The Ackley function"""

    def __init__(
        self, dim: int, a: Scalar = 20.0, b: Scalar = 0.2, c: Scalar = 2 * np.pi
    ):
        self.a = Scalar(a)
        self.b = Scalar(b)
        self.c = Scalar(c)
        super().__init__(dim=dim)

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.sum(x**2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(self.c * x)) / self.dim)
        return Scalar(term1 + term2 + self.a + np.exp(1))


class DropWave(Function):
    """The Drop-Wave function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return -1.0

    def eval(self, x: Vector) -> Scalar:
        r2 = sum(x**2)
        r = np.sqrt(r2)
        return Scalar(-(1 + np.cos(12 * r)) / (0.5 * r2 + 2))


class Eggholder(Function):
    """The Eggholder function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector([512.0, 404.2319])

    @property
    def f_star(self) -> Scalar:
        return -959.6407

    def eval(self, x: Vector) -> Scalar:
        a = x[1] + 47
        term1 = -a * np.sin(np.sqrt(abs(x[0] / 2 + a)))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - a)))
        return Scalar(term1 + term2)


class Griewank(Function):
    """The Griewank function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return Scalar(sum_term - prod_term + 1)


class Levy(Function):
    """The Levy function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.ones(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        term2 = sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        return Scalar(term1 + term2 + term3)


class Levy13(Function):
    """The Levy 13 function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector(np.ones(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        term1 = np.sin(3 * np.pi * x[0]) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
        return Scalar(term1 + term2 + term3)


class Schwefel(Function):
    """The Schwefel function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.full(self.dim, 420.9687))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return Scalar(418.9829 * self.dim - sum(x * np.sin(np.sqrt(abs(x)))))


class Booth(Function):
    """The Booth function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector([1.0, 3.0])

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return Scalar((x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2)


class Beale(Function):
    """The Beale function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector([3.0, 0.5])

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return Scalar(term1 + term2 + term3)


class Matyas(Function):
    """The Matyas function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return Scalar(0.26 * sum(x**2) - 0.48 * x[0] * x[1])


class SumSquares(Function):
    """The Sum Squares function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return sum((i + 1) * x[i] ** 2 for i in range(len(x)))


class Zakharov(Function):
    """The Zakharov function"""

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        sum1 = sum(x**2)
        sum2 = sum(0.5 * (i + 1) * x[i] for i in range(len(x)))
        return sum1 + sum2**2 + sum2**4


class ThreeHumpCamel(Function):
    """The Three-Hump Camel function"""

    def __init__(self):
        super().__init__(dim=2)

    @property
    def x_star(self) -> Vector:
        return Vector(np.zeros(self.dim, dtype=dtype))

    @property
    def f_star(self) -> Scalar:
        return 0.0

    def eval(self, x: Vector) -> Scalar:
        return (
            2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2
        )
