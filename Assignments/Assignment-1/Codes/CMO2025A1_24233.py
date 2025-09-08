# ---------- CMO 2025 Assignment 1 ----------

# ---------- Imports ----------
# Allowed libraries: `os`, `sys`, `numpy`, `math`, `matplotlib`.
# Additional allowed: `time``, `tqdm`-like.

import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.traceback import install

sys.path.insert(0, os.path.abspath("oracle_2025A1"))
from oracle_2025A1 import oq1, oq2f, oq2g, oq3  # type: ignore

install()  # For rich tracebacks in case of errors
console = Console()

# Type Aliases
floatVec = npt.NDArray[np.float64]
"""A type alias for a numpy array of real numbers (i.e., float)."""


# ---------- Setup ----------
SRN: int = 24233
"""The 5-digit Student Registration Number (SRN) for the assignment."""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."

## Enable/disable optional configurations
ORACLE_CACHE: bool = False
"""Cache the results of the oracle calls."""

LOG_RUNS: bool = True
"""Log all the runs of the optimisation algorithms in the summary table, not just the best one."""

PLOT_CONVERGENCE: bool = True
"""Plot the convergence of the optimisation algorithms over iterations."""

np.random.seed(SRN)


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

    @classmethod
    def from_separate(cls, f_fn, grad_fn, dim: int, cache_digits: int = 32):
        """Construct an oracle from separate f(x) and f'(x) functions."""

        def oracle(srn: int, x: floatVec) -> tuple[float, floatVec]:
            return f_fn(srn, x), grad_fn(srn, x)

        return cls(oracle, dim=dim, cache_digits=cache_digits)


# ---------- Algorithm Templates ----------
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

        self.history: list[floatVec] = []
        self.x_star: floatVec
        self.fx_star: float
        self.dfx_star: floatVec

        self.maxiter: int
        self.tol: float

    def _initialise_state(self) -> None:
        """
        Initialises the state of the algorithm.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    def _step(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> floatVec:
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
        x0s: list[floatVec],
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

        # Summary table for multiple starting points
        def format_array(arr, precision=6):
            return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"

        console = Console()
        table = Table(title=f"{self.name}", show_lines=True)
        cols = ["Run", "x0", "x*", "f(x*)", "f'(x*)", "Iterations", "Oracle calls"]
        for col in cols:
            table.add_column(col, justify="center")

        self.runs = []
        with Live(table, console=console, transient=True) as live:
            for idx, x0 in enumerate(x0s, start=1):
                oracle_fn.reset()
                history = [x0]
                x = x0
                self._initialise_state()

                try:
                    for k in range(1, maxiter + 1):
                        fx, dfx = oracle_fn(x)  # Query the oracle function
                        if np.linalg.norm(dfx) < tol:
                            # Early exit, ||f'(x)|| is small enough
                            break
                        x = self._step(x, k, fx, dfx, oracle_fn)
                        history.append(x)
                    fx, dfx = oracle_fn(x)
                except OverflowError:  # Fallback
                    x = np.full(oracle_fn.dim, np.nan)
                    fx, dfx = float("nan"), np.full(oracle_fn.dim, np.nan)

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
                    table.add_row(
                        str(idx),
                        format_array(x0),
                        format_array(x),
                        f"{fx:2.16f}",
                        format_array(dfx),
                        str(len(history) - 1),
                        str(oracle_fn.call_count),
                    )
                    live.refresh()

        # Pick best run by lowest ||f'(x^*)||, if tied then prefer lower oracle call count
        if valid_runs := [
            r
            for r in self.runs
            if not (math.isnan(r["fx_star"]) or np.any(np.isnan(r["dfx_star"])))
        ]:
            best = min(
                valid_runs,
                key=lambda r: (np.linalg.norm(r["dfx_star"]), r["oracle_call_count"]),
            )
            run_idx = next(i for i, run in enumerate(self.runs, start=1) if run is best)
            self.x_star = best["x_star"]
            self.fx_star, self.dfx_star = best["fx_star"], best["dfx_star"]
            self.history = best["history"]
            x0 = best["x0"]
            n_iters = len(best["history"]) - 1
            n_oracle = best["oracle_call_count"]
        else:
            run_idx = ""
            self.x_star = np.full(oracle_fn.dim, np.nan)
            self.fx_star, self.dfx_star = float("nan"), np.full(oracle_fn.dim, np.nan)
            self.history = []
            x0 = np.full(oracle_fn.dim, np.nan)
            n_iters = ""
            n_oracle = ""

        table.add_row(
            str(run_idx),
            format_array(x0),
            format_array(self.x_star),
            f"{self.fx_star:2.16f}",
            format_array(self.dfx_star),
            str(n_iters),
            str(n_oracle),
            style="bold magenta",
        )
        if LOG_RUNS:
            console.print(table)

    def plot(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)


class LineSearchOptimiser(IterativeOptimiser):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + eta_k * d_k`\\
    where `eta_k` is the step size along the descent direction `d_k`.
    """

    def _initialise_state(self):
        super()._initialise_state()
        self.step_sizes: list[float] = []

    def _direction(self, x: floatVec, grad: floatVec) -> floatVec:
        """
        Returns the descent direction `d_k` to move towards from `x_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific direction strategy.
        """
        raise NotImplementedError

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        """Returns step size `eta_k` to take along the descent direction `d_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step size strategy.
        """
        raise NotImplementedError

    def _step(self, x, k, f, grad, oracle_fn):
        d_k = self._direction(x, grad)
        eta_k = self._step_size(x, k, f, grad, d_k, oracle_fn)
        self.step_sizes.append(eta_k)
        return x + eta_k * d_k

    def plot_step_sizes(self):
        """Plot step sizes vs iterations for the best run."""
        plt.plot(self.step_sizes, marker="o", label=self.name)


class SteepestDescentDirectionMixin(LineSearchOptimiser):
    """
    A mixin class that provides the steepest descent direction strategy.

    `d_k = -f'(x_k)`
    """

    def _direction(self, x: floatVec, grad: floatVec) -> floatVec:
        return -grad


class ExactLineSearchMixin(LineSearchOptimiser):
    """
    A mixin class that provides the exact line search step size strategy for quadratic functions.

    `eta_k = - (f'(x_k)^T d_k) / (d_k^T Q d_k)`\\
    where `Q` is the Hessian matrix of the quadratic function.
    """

    def _initialise_state(self):
        super()._initialise_state()
        Q = self.config.get("Q", None)
        if Q is None:
            raise ValueError("Q matrix is required for exact line search.")
        self.Q: floatVec = np.array(Q, dtype=float)

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        numer = float(grad.T @ direction)
        denom = float(direction.T @ self.Q @ direction)
        if abs(denom) < 1e-14:
            return 1e-8
        return -numer / denom


# ---------- Optimiser Implementations ----------
class SteepestGradientDescentExactLineSearch(
    SteepestDescentDirectionMixin, ExactLineSearchMixin, LineSearchOptimiser
):
    """
    Steepest Gradient Descent with Exact Line Search for Quadratic Functions.

    `x_{k+1} = x_k - eta_k * f'(x_k)`\\
    where `eta_k = (f'(x_k)^T f'(x_k)) / (f'(x_k)^T Q f'(x_k))`
    """


class SteepestGradientDescentArmijo(SteepestDescentDirectionMixin, LineSearchOptimiser):
    """
    Steepest Gradient Descent using (Inexact) Line Search with Armijo condition.

    `f(x_k + eta_k * d_k) <= f(x_k) + alpha * eta_k * f'(x_k)^T d_k`
    """

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        alpha = float(self.config.get("alpha", 0.3))
        beta = float(self.config.get("beta", 0.8))
        eta = float(self.config.get("initial_step_size", 1.0))

        while True:
            new_x = x + eta * direction
            new_f, _ = oracle_fn(new_x)
            if new_f <= f + alpha * eta * (grad.T @ direction):
                break
            eta *= beta
            if eta < 1e-14:
                eta = 1e-14
                break
        return eta


class SteepestGradientDescentArmijoGoldstein(
    SteepestDescentDirectionMixin, LineSearchOptimiser
):
    """
    Steepest Gradient Descent using (Inexact) Line Search with Armijo-Goldstein condition.

    `f(x_k + eta_k * d_k) <= f(x_k) + alpha * eta_k * f'(x_k)^T d_k`\\
    `f(x_k + eta_k * d_k) >= f(x_k) + (1 - alpha) * eta_k * f'(x_k)^T d_k`\\
    where `0 < alpha < 0.5`.
    """

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        alpha = float(self.config.get("alpha", 0.3))
        beta = float(self.config.get("beta", 0.8))
        eta = float(self.config.get("initial_step_size", 1.0))

        while True:
            new_x = x + eta * direction
            new_f, _ = oracle_fn(new_x)
            if new_f <= f + alpha * eta * (grad.T @ direction) and new_f >= f + (
                1 - alpha
            ) * eta * (grad.T @ direction):
                break
            eta *= beta
            if eta < 1e-14:
                eta = 1e-14
                break
        return eta


class SteepestGradientDescentWolfe(SteepestDescentDirectionMixin, LineSearchOptimiser):
    """
    Steepest Gradient Descent using (Inexact) Line Search with Wolfe condition.

    `f(x_k + eta_k * d_k) <= f(x_k) + alpha * eta_k * f'(x_k)^T d_k`\\
    `f'(x_k + eta_k * d_k)^T d_k >= beta * f'(x_k)^T d_k`\\
    where `0 < alpha < beta < 1`.
    """

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        alpha = float(self.config.get("alpha", 0.3))
        beta = float(self.config.get("beta", 0.8))
        eta = float(self.config.get("initial_step_size", 1.0))

        while True:
            new_x = x + eta * direction
            new_f, new_grad = oracle_fn(new_x)
            if new_f <= f + alpha * eta * (grad.T @ direction) and (
                new_grad.T @ direction
            ) >= beta * (grad.T @ direction):
                break
            eta *= 0.5
            if eta < 1e-14:
                eta = 1e-14
                break
        return eta


class SteepestGradientDescentBacktracking(
    SteepestDescentDirectionMixin, LineSearchOptimiser
):
    """
    Steepest Gradient Descent using (Inexact) Line Search with Backtracking.

    `f(x_k + eta_k * d_k) <= f(x_k) + alpha * eta_k * f'(x_k)^T d_k`\\
    where `0 < alpha < 0.5`.
    """

    def _step_size(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        alpha = float(self.config.get("alpha", 0.3))
        beta = float(self.config.get("beta", 0.8))
        eta = float(self.config.get("initial_step_size", 1.0))

        while True:
            new_x = x + eta * direction
            new_f, _ = oracle_fn(new_x)
            if new_f <= f + alpha * eta * (grad.T @ direction):
                break
            eta *= beta
            if eta < 1e-14:
                eta = 1e-14
                break
        return eta


# ---------- Utils ----------
class LinearSystem:
    """
    A class to represent and solve a linear system of equations `Ax = b`.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

        self.m, self.n = A.shape
        assert self.m == b.shape[0], "A and b must have compatible dimensions."

    def solve(self) -> floatVec:
        """Solves the linear system `Ax = b` using numpy's solver."""
        return np.linalg.solve(self.A, self.b)

    def residual(self, x: floatVec) -> floatVec:
        """Computes the residual `Ax - b` for a given `x`."""
        return self.A @ x - self.b

    def make_oracle(self) -> FirstOrderOracle:
        """
        Creates a first-order oracle for the least squares problem.

        `f(x) = 0.5 * ||Ax - b||^2`\\
        `f'(x) = A^T (Ax - b)`
        """

        def oracle(_srn: int, x: floatVec) -> tuple[float, floatVec]:
            r = self.residual(x)
            f = float(0.5 * r.T @ r)
            grad = self.A.T @ r
            return f, grad

        return FirstOrderOracle(oracle, dim=self.n)


# ---------- Questions ----------
def question_1():
    console.rule("[bold green]Question 1")

    b = np.array([1, 1], dtype=float)
    for i, Q in enumerate(oq1(SRN)):
        if i != 0:
            console.rule(style="default")
        Q_name = "Q_" + chr(ord("a") + i)
        print(" " * 7 + "\u250c" + " " * 28 + "\u2510")
        print(f"{Q_name:^5}= \u2502", end="")
        print(" ".join(f"{val:13.8f}" for val in Q[0]) + " \u2502")
        print(" " * 7 + "\u2502", end="")
        print(" ".join(f"{val:13.8f}" for val in Q[1]) + " \u2502")
        print(" " * 7 + "\u2514" + " " * 28 + "\u2518")

        oracle = FirstOrderOracle(
            lambda _, x: (0.5 * x.T @ Q @ x + b.T @ x, Q @ x + b), dim=2
        )
        optim = SteepestGradientDescentExactLineSearch(Q=Q)
        x0s = [np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
        optim.run(oracle, x0s=x0s, maxiter=1_000, tol=1e-13)

        x_star_analytical = -np.linalg.solve(Q, b)
        fx_star_analytical, dfx_star_analytical = oracle(x_star_analytical)
        console.print("[bold green]Analytical solution:[/]")
        print(f"x* = {x_star_analytical}")
        print(f"f(x*) = {fx_star_analytical:2.16f}")
        print(f"f'(x*) = {dfx_star_analytical}")

        # Plot ||x^(k) - x^*|| for the best run
        xk_history = np.array(optim.history)
        norm_diff = np.linalg.norm(xk_history - x_star_analytical, axis=1)
        plt.figure()
        plt.plot(norm_diff, marker="o")
        plt.title(r"$\|x^{(k)} - x^*\|$ vs Iteration $k$" + f" for {Q_name}")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"$\|x^{(k)} - x^*\|$")
        plt.grid(True)


def question_2():
    console.rule("[bold green]Question 2")

    def grad_fn(srn: int, x: floatVec) -> floatVec:
        g: floatVec = oq2g(srn, x)
        return g.reshape(-1)

    oracle = FirstOrderOracle.from_separate(oq2f, grad_fn, dim=5)

    optimisers: list[LineSearchOptimiser] = [
        SteepestGradientDescentArmijo(alpha=0.3, beta=0.8, initial_step_size=1.0),
        SteepestGradientDescentArmijoGoldstein(
            alpha=0.3, beta=0.8, initial_step_size=1.0
        ),
        SteepestGradientDescentWolfe(alpha=0.3, beta=0.8, initial_step_size=1.0),
        SteepestGradientDescentBacktracking(alpha=0.3, beta=0.8, initial_step_size=1.0),
    ]
    x0s = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
    ]
    plt.figure()
    for optim in optimisers:
        optim.run(oracle, x0s=x0s, maxiter=1_000, tol=1e-7)
        optim.plot_step_sizes()
    plt.title(r"Step Size vs Iteration for different (Inexact) Line Search methods")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Step Size $\eta_k$")
    plt.legend()
    plt.grid(True)


def question_3():
    console.rule("[bold green]Question 3")

    question_3_2()
    question_3_5()


def question_3_2():
    t0 = time.perf_counter()
    A, b = oq3(SRN)
    t1 = time.perf_counter()
    console.print(
        f"[bright_black]Time taken to load A, b from oracle: {t1 - t0:.6f} seconds[/]"
    )
    A: floatVec = np.array(A, dtype=np.float64)
    b: floatVec = np.array(b.reshape(-1), dtype=np.float64)
    print(f"A.shape: {A.shape}")
    print(f"b.shape: {b.shape}")

    ls = LinearSystem(A, b)
    oracle = ls.make_oracle()


def question_3_5():
    MU = 0
    SIGMA = 1

    m_values = [2**i for i in range(1, 13)]
    for m in m_values:
        try:
            console.print(f"[cyan]Generating LinearSystem [italic]for {m=}[/]")
            t0 = time.perf_counter()
            A: floatVec = SIGMA * np.random.randn(m, m) + MU
            b: floatVec = SIGMA * np.random.randn(m) + MU
            t1 = time.perf_counter()
            console.print(
                f"[bright_black]Time taken to generate A, b: {t1 - t0:.6f} seconds[/]"
            )
            ls = LinearSystem(A, b)

            t0 = time.perf_counter()
            _sol = -np.linalg.inv(A) * b
            t1 = time.perf_counter()
            console.print(
                f"[bright_black]Time taken to solve system using matrix inversion: {t1 - t0:.6f} seconds[/]"
            )

            t0 = time.perf_counter()
            ls.solve()
            t1 = time.perf_counter()
            console.print(
                f"[bright_black]Time taken to solve system using numpy.linalg: {t1 - t0:.6f} seconds[/]"
            )

        except MemoryError as e:
            console.print(
                f"[yellow][bold]MemoryError:[/bold] Unable to allocate [italic]for {m=}:[/italic][/yellow] {e}"
            )
        except Exception as e:
            console.print(
                f"[red][bold]Error[/bold] [italic]for {m=}:[/italic][/red] {e}"
            )


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()

    plt.show()
