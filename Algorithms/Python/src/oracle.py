"""
oracle.py
=======
Oracle utils
"""

import numpy as np

from .types import floatVec

ORACLE_CACHE: bool = False
"""Cache the results of the oracle calls."""


class FirstOrderOracle:
    """
    A wrapper class around the provided `oracle` function.

    `f(x), f'(x) = oracle(SRN, x)`
    """

    def __init__(self, oracle, dim: int, cache_digits: int = 32):
        self.oracle = oracle
        """
        The oracle function to be wrapped.
        It should be of the form `f(x), f'(x) = oracle(SRN, x)`.
        """

        self.dim = dim
        """Dimension of the input `x` for the oracle function."""

        self.call_count: int = 0
        """
        Tracks the number of times the oracle function has been called.
        This is useful for the 'analytical complexity' of the algorithms.
        """

        self.cache: dict[tuple[float], tuple[float, floatVec]] = {}
        """Cache the results of the oracle function."""

        self.cache_digits = cache_digits
        """
        Number of digits to round `x` for caching results.
        This is useful to avoid floating-point precision issues when caching results.
        """

    def __call__(self, x: floatVec) -> tuple[float, floatVec]:
        """Evaluates the oracle function at `x`, using cache if available."""
        x = np.asarray(x, dtype=float)
        assert x.shape == (self.dim,), f"x must be of shape ({self.dim},)"

        # Round for stable caching, and hash as a tuple
        x_key = tuple(np.round(x, self.cache_digits))
        if ORACLE_CACHE:
            if x_key in self.cache:
                return self.cache[x_key]

        fx, dfx = self.oracle(x)
        self.call_count += 1

        if ORACLE_CACHE:
            self.cache[x_key] = (fx, dfx)

        return fx, dfx

    def reset(self) -> "FirstOrderOracle":
        """Resets the internal call count and cache."""
        self.call_count = 0
        self.cache.clear()
        return self

    @classmethod
    def from_separate(cls, f_fn, grad_fn, dim: int, cache_digits: int = 32):
        """Construct an oracle from separate f(x) and f'(x) functions."""

        def oracle(srn: int, x: floatVec) -> tuple[float, floatVec]:
            return f_fn(srn, x), grad_fn(srn, x)

        return cls(oracle, dim=dim, cache_digits=cache_digits)


class ConvexQuadraticOracle(FirstOrderOracle):
    """
    A wrapper oracle for convex quadratic functions of the form:

    `f(x) = 0.5 * x^T Q x + b^T x`\\
    `f'(x) = Q x + b`\\
    where `Q` is a symmetric positive definite matrix and `b` is a vector.
    """

    def __init__(self, Q: floatVec, b: floatVec):
        assert Q.shape[0] == Q.shape[1], "Q must be a square matrix."
        assert Q.shape[0] == b.shape[0], "Dimensions of Q and b must match."

        # Check for symmetric positive definite
        if not np.allclose(Q, Q.T) or np.any(np.linalg.eigvals(Q) <= 0):
            raise ValueError("Q must be a symmetric positive definite matrix.")

        self.Q: floatVec = Q
        self.b: floatVec = b
        super().__init__(self._oracle_fn, dim=Q.shape[1])

    def _oracle_fn(self, _: int, x: floatVec) -> tuple[float, floatVec]:
        """The oracle function for the convex quadratic."""
        fx = float(0.5 * x.T @ self.Q @ x + self.b.T @ x)
        dfx: floatVec = self.Q @ x + self.b
        return fx, dfx
