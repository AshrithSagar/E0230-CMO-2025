# ---------- CMO 2025 Assignment 1 ----------

# ---------- Imports ----------
# Allowed libraries: os, sys, numpy, math, matplotlib.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("oracle_2025A1"))
from oracle_2025A1 import oq1, oq2f, oq2g, oq3  # type: ignore

# ---------- Setup ----------
SRN: int = 24233
"""The 5-digit Student Registration Number (SRN) for the assignment."""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."

## Enable/disable optional configurations
ORACLE_CACHE: bool = False
"""Cache the results of the oracle calls."""


# ---------- Oracle utils ----------
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

        self.cache: dict[np.ndarray, tuple[np.ndarray, np.ndarray]] = {}
        """Cache the results of the oracle function."""

        self.cache_digits = cache_digits
        """
        Number of digits to round `x` for caching results.
        This is useful to avoid floating-point precision issues when caching results.
        """

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the oracle function at `x`, using cache if available."""
        x = np.asarray(x, dtype=float)
        assert x.shape == (self.dim,), f"x must be of shape ({self.dim},)"

        x_key = np.round(x, self.cache_digits)  # Round for stable caching
        if ORACLE_CACHE:
            if x_key in self.cache:
                return self.cache[x_key]

        fx, dfx = self.oracle(SRN, x)
        self.call_count += 1

        if ORACLE_CACHE:
            self.cache[x_key] = (fx, dfx)

        return fx, dfx

    def reset(self) -> "FirstOrderOracle":
        """Resets the internal call count and cache."""
        self.call_count = 0
        self.cache.clear()
        return self


# ---------- Questions ----------
def question_1():
    print("\n" + "\033[1m\033[4m" + "Question 1" + "\033[0m")

    Q_a, Q_b, Q_c, Q_d, Q_e = oq1(SRN)  # noqa: F841
    for i in range(5):
        name = f"Q_{chr(ord('a') + i)}"
        Q = locals()[name]
        print(" " * 7 + "\u250c" + " " * 28 + "\u2510")
        print(f"{name:^5}= \u2502", end="")
        print(" ".join(f"{val:13.8f}" for val in Q[0]) + " \u2502")
        print(" " * 7 + "\u2502", end="")
        print(" ".join(f"{val:13.8f}" for val in Q[1]) + " \u2502")
        print(" " * 7 + "\u2514" + " " * 28 + "\u2518")


def question_2():
    print("\n" + "\033[1m\033[4m" + "Question 2" + "\033[0m")

    oracle_f = FirstOrderOracle(oq2f, dim=5)  # noqa: F841
    oracle_g = FirstOrderOracle(oq2g, dim=5)  # noqa: F841


def question_3():
    print("\n" + "\033[1m\033[4m" + "Question 3" + "\033[0m")

    A, b = oq3(SRN)
    print(f"A.shape: {A.shape}")
    print(f"b.shape: {b.shape}")


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")
    question_1()
    question_2()
    question_3()
