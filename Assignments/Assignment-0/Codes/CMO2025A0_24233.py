# ---------- CMO 2025 Assignment 0 ----------

# ---------- Imports ----------
# Allowed libraries: os, sys, numpy, math, matplotlib.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

# ---------- Setup ----------
SRN: int = 24233
"""The 5-digit Student Registration Number (SRN) for the assignment."""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."

## Enable/disable optional configurations
ORACLE_PLOT: bool = False
"""Plot the graph of `f(x)` by querying the oracle function."""

ORACLE_CACHE: bool = False
"""Cache the results of the oracle calls."""

PLOT_CONVERGENCE: bool = False
"""Plot the convergence of the optimisation algorithms over iterations."""

LOG_RUNS: bool = True
"""Log all the runs of the optimisation algorithms in the summary table, not just the best one."""


# ---------- Oracle utils ----------
class FirstOrderOracle:
    """
    A wrapper class around the provided `oracle` function.

    `f(x), f'(x) = oracle(SRN, x)`
    """

    def __init__(self, cache_digits: int = 32):
        self.call_count: int = 0
        """
        Tracks the number of times the oracle function has been called.
        This is useful for the 'analytical complexity' of the algorithms.
        """

        self.cache: dict[float, tuple[float, float]] = {}
        """Cache the results of the oracle function."""

        self.cache_digits = cache_digits
        """
        Number of digits to round `x` for caching results.
        This is useful to avoid floating-point precision issues when caching results.
        """

    def __call__(self, x: float) -> tuple[float, float]:
        """Evaluates the oracle function at `x`, using cache if available."""
        if ORACLE_CACHE:
            x_key = round(x, self.cache_digits)  # Round for stable caching
            if x_key in self.cache:
                return self.cache[x_key]

        fx, dfx = oracle(SRN, x)
        self.call_count += 1

        if ORACLE_CACHE:
            self.cache[x] = (fx, dfx)

        return fx, dfx

    def reset(self) -> "FirstOrderOracle":
        """Resets the internal call count and cache."""
        self.call_count = 0
        self.cache.clear()
        return self

    def plot(self, x_range: tuple[float, float], num_points: int | None = None):
        """
        Plots the oracle function over a specified range.\\
        This is just a convenience method to visualise the function `f(x)`
        as this queries the oracle multiple times, which may not be desirable in practice.

        Parameters:
            x_range: A tuple specifying the range of `x` values to plot.
            num_points: Number of points to sample in the range.
        """
        lower, upper = x_range
        if num_points is None:
            num_points = int(upper - lower + 1)
        x_values = np.linspace(lower, upper, num_points)
        y_values = [self.__call__(x)[0] for x in x_values]

        plt.plot(x_values, y_values, label="f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Oracle Function Plot")
        plt.legend()
        plt.grid(True)


# ---------- Iterative Algorithm Template ----------
class IterativeOptimiser:
    """
    A base template class for iterative optimisation algorithms,
    particularly used here for the minimisation objective.

    `x_{k+1} = ALGO(x_k)`\\
    where `ALGO` is the algorithm-specific step function,
    `x_k` is the value at iteration `k`.
    """

    def __init__(self, **kwargs):
        # Initialises the iterative optimiser with configuration parameters.
        self.config = kwargs

        self.name = self.__class__.__name__
        """Name of the algorithm, derived from the class name of the optimiser."""

        self.history: list[float] = []
        self.x_star: float
        self.fx_star: float
        self.dfx_star: float

        self.maxiter: int
        self.tol: float

    def _initialise_state(self) -> None:
        """
        Initialises the state of the algorithm.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    def _step(
        self, x: float, k: int, f: float, grad: float, oracle_fn: FirstOrderOracle
    ) -> float:
        """
        Performs a single step of the algorithm.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            x: Current value of `x`, i.e., `x_k`.
            k: Current iteration number.
            f: Current function value `f(x)`, viz. `f(x_k)`.
            grad: Current gradient `f'(x)`, viz. `f'(x_k)`.
            oracle_fn: The oracle function to query for `f(x)` and `f'(x)`.
        Returns:
            The updated value of `x` after the step, viz. `x_{k+1}`.
        """
        raise NotImplementedError

    def run(
        self,
        oracle_fn: FirstOrderOracle,
        x0s: list[float],
        maxiter: int = 1_000_000,
        tol: float = 1e-9,
    ):
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0s: Initial guesses for the minimum point.
            maxiter: Maximum number of iterations to perform.
            tol: Tolerance for stopping criterion based on the gradient.
        """
        self.maxiter = maxiter
        self.tol = tol

        # Set oracle_fn's cache_digits in order of tol
        oracle_fn.cache_digits = int(-np.log10(tol)) + 1

        # Summary table
        cols = [
            ("Run", "{:^5}", 5, "{}"),
            ("x0", "{:>10} ", 11, "{:2.6f}"),
            ("x*", "{:>20} ", 21, "{:2.16f}"),
            ("f(x*)", "{:>20} ", 21, "{:2.16f}"),
            ("f'(x*)", "{:>13} ", 14, "{:.6e}"),
            ("Iterations", "{:^12}", 12, "{}"),
            ("Oracle calls", "{:^14}", 14, "{}"),
        ]
        width = sum(c[2] for c in cols) + len(cols) + 1
        print("\n" + self.name.center(width))
        row_format = "\u2502" + "\u2502".join(c[1] for c in cols) + "\u2502"
        print("\u250f" + "\u2533".join("\u2501" * c[2] for c in cols) + "\u2513")
        print(
            row_format.replace(">", "^")
            .replace("\u2502", "\u2503")
            .format(*(c[0] for c in cols))
        )
        print("\u2521" + "\u2547".join("\u2501" * c[2] for c in cols) + "\u2529")

        self.runs = []
        for idx, x0 in enumerate(x0s, start=1):
            oracle_fn.reset()
            history = [x0]
            x = x0
            self._initialise_state()

            try:
                for k in range(1, maxiter + 1):
                    fx, dfx = oracle_fn(x)  # Query the oracle function
                    if abs(dfx) < tol:  # Early exit if f'(x) is small enough
                        break
                    x = self._step(x, k, fx, dfx, oracle_fn)
                    history.append(x)
                fx, dfx = oracle_fn(x)
            except OverflowError:  # Fallback
                x = float("nan")
                fx, dfx = float("nan"), float("nan")

            self.runs.append(
                {
                    "x0": x0,
                    "x_star": x,
                    "fx_star": fx,
                    "dfx_star": dfx,
                    "history": history,
                    "oracle_call_count": oracle_fn.call_count,
                }
            )

            if LOG_RUNS:
                row = [idx, x0, x, fx, dfx, len(history) - 1, oracle_fn.call_count]
                print(row_format.format(*[c[3].format(v) for c, v in zip(cols, row)]))
        if LOG_RUNS:
            print("\u251c" + "\u253c".join("\u2500" * c[2] for c in cols) + "\u2524")

        # Pick best run by lowest abs(f'(x^*)), if tied then lower oracle call count
        best = min(
            self.runs, key=lambda r: (abs(r["dfx_star"]), r["oracle_call_count"])
        )
        self.x_star = best["x_star"]
        self.fx_star = best["fx_star"]
        self.dfx_star = best["dfx_star"]
        self.history = best["history"]

        row = [
            self.runs.index(best) + 1,
            best["x0"],
            best["x_star"],
            best["fx_star"],
            best["dfx_star"],
            len(best["history"]) - 1,
            best["oracle_call_count"],
        ]
        print(row_format.format(*[c[3].format(v) for c, v in zip(cols, row)]))
        print("\u2514" + "\u2534".join("\u2500" * c[2] for c in cols) + "\u2518")

    def plot(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)


# ---------- Optimiser Implementations ----------
class GradientDescent(IterativeOptimiser):
    """
    Standard Gradient Descent.

    `x_{k+1} = x_k - eta f'(x_k)`\\
    where `eta` is the learning rate.
    """

    def _step(self, x, k, f, grad, oracle_fn):
        eta: float = self.config["lr"]
        return x - eta * grad


class BacktrackingGradientDescent(IterativeOptimiser):
    """
    Gradient Descent with Backtracking Line Search.

    Starts with an initial learning rate and reduces it using Armijo condition:
    `f(x - eta * grad) <= f(x) - alpha * eta * grad^2`
    where `eta` is the step size, `alpha` is a constant, and `beta` is the reduction factor.
    The step size is reduced until the condition is satisfied.
    """

    def _step(self, x, k, f, grad, oracle_fn):
        eta = self.config["init_lr"]
        alpha = self.config["alpha"]
        beta = self.config["beta"]

        f_x, _ = oracle_fn(x)
        while True:
            x_new = x - eta * grad
            f_new, _ = oracle_fn(x_new)  # Needs multiple queries

            # Armijo condition
            if f_new <= f_x - alpha * eta * grad**2:
                break  # Step size is good enough
            eta *= beta  # Reduce step size

        return x - eta * grad  # Step


class BFGS(IterativeOptimiser):
    """
    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm
    which approximates the inverse Hessian.\\
    This is a quasi-Newton algorithm, and the 1D case is implemented here.

    `x_{k+1} = x_k - H_k * f'(x_k)`\\
    where `H_k` is the inverse Hessian approximation at iteration `k`.

    `H_init` is the initial inverse Hessian approximation.
    """

    def _initialise_state(self):
        self.H = self.config.get("H_init", 1.0)
        self.prev_grad = None
        self.prev_x = None

    def _step(self, x, k, f, grad, oracle_fn):
        if self.prev_x is not None and self.prev_grad is not None:
            s = x - self.prev_x
            y = grad - self.prev_grad
            denom = s * y

            if denom != 0:  # Scalar update in 1D
                self.H = (s**2) / denom

        # Update current values
        self.prev_x = x
        self.prev_grad = grad

        return x - self.H * grad  # Step


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    oracle_f = FirstOrderOracle()
    if ORACLE_PLOT:
        oracle_f.plot(x_range=(-100, 100))

    x0s = np.linspace(-1, 1, 11).tolist()
    print(f"Initial points: [{', '.join(f'{x:.3f}' for x in x0s)}]")

    optimisers: list[IterativeOptimiser] = [
        GradientDescent(lr=0.2),
        BacktrackingGradientDescent(init_lr=0.5, alpha=0.1, beta=0.9),
        BFGS(H_init=0.5),
    ]
    for opt in optimisers:
        opt.run(oracle_f, x0s=x0s, maxiter=1_000_000, tol=1e-13)

    # Convergence of optimisers plot
    if PLOT_CONVERGENCE:
        plt.figure()
        for opt in optimisers:
            opt.plot()
        plt.xlabel("Iteration")
        plt.ylabel("x value")
        plt.title("Convergence of Optimisation Algorithms")
        plt.legend()
        plt.grid(True)

    plt.show()
