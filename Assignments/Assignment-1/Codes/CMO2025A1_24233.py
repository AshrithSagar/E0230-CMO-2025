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
from rich.progress import Progress, TextColumn, TimeElapsedColumn
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

LOG_RUNS: bool = False
"""Log all the runs of the optimisation algorithms in the summary table, not just the best one."""

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

    def initialise_state(self) -> None:
        """
        Initialises the state of the algorithm.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    def step(
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
        show_params: bool = True,
    ):
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0s: Initial guesses for the minimum point.
            maxiter: Maximum number of iterations to perform.
            tol: Tolerance for stopping criterion based on the gradient.
            show_params: Whether to display the configuration parameters of the algorithm.
        """
        self.maxiter = maxiter
        self.tol = tol

        # Set oracle_fn's cache_digits in order of tol
        oracle_fn.cache_digits = int(-np.log10(tol)) + 1

        console = Console()
        console.print(f"[bold blue]{self.name}[/]")
        if show_params:
            console.print(f"params: {self.config}")
        has_multiple_x0 = len(x0s) > 1

        self.runs = []
        for idx, x0 in enumerate(x0s, start=1):
            oracle_fn.reset()
            history = [x0]
            x = x0
            self.initialise_state()

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn("iter:{task.completed:04},"),
                TextColumn("f(x):{task.fields[fx]:.4e},"),
                TextColumn("||f'(x)||: {task.fields[grad_norm]:.2e},"),
                TextColumn("Oracle calls: {task.fields[oracle_calls]:04}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
            progress.start()
            try:
                task = progress.add_task(
                    "Run" + f" {idx}" if has_multiple_x0 else "" + ":",
                    total=maxiter,
                    fx=float("nan"),
                    grad_norm=float("nan"),
                    oracle_calls=0,
                )
                for k in range(1, maxiter + 1):
                    fx, dfx = oracle_fn(x)  # Query the oracle function
                    grad_norm = np.linalg.norm(dfx)
                    progress.update(
                        task,
                        advance=1,
                        fx=fx,
                        grad_norm=grad_norm,
                        oracle_calls=oracle_fn.call_count,
                    )
                    if grad_norm < tol:  # Early exit, if ||f'(x)|| is small enough
                        break
                    x = self.step(x, k, fx, dfx, oracle_fn)
                    history.append(x)
                fx, dfx = oracle_fn(x)
            except OverflowError:  # Fallback, in case of non-convergence
                x = np.full(oracle_fn.dim, np.nan)
                fx, dfx = float("nan"), np.full(oracle_fn.dim, np.nan)
            finally:
                progress.stop()

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
            if LOG_RUNS and has_multiple_x0:
                console.print(f"\n[bold yellow]Run {idx}:[/]")
                print(f"x0 = {format_float(x0, sep=', ')}")
                print(f"Iterations = {len(history) - 1}")
                print(f"Oracle calls = {oracle_fn.call_count}")
                print(f"x* = {format_float(x, sep=', ')}")
                print(f"f(x*) = {format_float(fx)}")
                print(f"f'(x*) = {format_float(dfx, sep=', ')}")
                print(f"||f'(x*)|| = {np.linalg.norm(dfx):g}")

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

        if has_multiple_x0:
            console.print("\n[bold green]Best run:[/]")
        print(f"x0 = {format_float(x0, sep=', ')}", end="")
        print(f", at index {run_idx}" if has_multiple_x0 else "")
        print(f"Iterations = {n_iters}")
        print(f"Oracle calls = {n_oracle}")
        print(f"x* = {format_float(self.x_star, sep=', ')}")
        print(f"f(x*) = {format_float(self.fx_star)}")
        print(f"f'(x*) = {format_float(self.dfx_star, sep=', ')}")
        print(f"||f'(x*)|| = {np.linalg.norm(self.dfx_star):g}")

    def plot_history(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)


class LineSearchOptimiser(IterativeOptimiser):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `alpha_k` is the step length along the descent direction `p_k`.
    """

    def initialise_state(self):
        super().initialise_state()
        self.step_lengths: list[float] = []

    def direction(self, x: floatVec, grad: floatVec) -> floatVec:
        """
        Returns the descent direction `p_k` to move towards from `x_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific direction strategy.
        """
        raise NotImplementedError

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        """
        Returns step length `alpha_k` to take along the descent direction `p_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step length strategy.
        """
        raise NotImplementedError

    def step(self, x, k, f, grad, oracle_fn):
        p_k = self.direction(x, grad)
        alpha_k = self.step_length(x, k, f, grad, p_k, oracle_fn)
        self.step_lengths.append(alpha_k)
        return x + alpha_k * p_k

    def plot_step_lengths(self):
        """Plot step lengths vs iterations for the best run."""
        plt.plot(self.step_lengths, marker="o", label=self.name)

    def _phi_and_derphi(
        self, x: floatVec, alpha: float, d: floatVec, oracle_fn: FirstOrderOracle
    ):
        """Computes\\
        `phi(alpha) = f(x + alpha * d)`\\
        `phi'(alpha) = f'(x + alpha * d)^T d`
        """
        xa = x + alpha * d
        fa, g_a = oracle_fn(xa)
        return fa, float(g_a.T @ d)


class SteepestDescentDirectionMixin(LineSearchOptimiser):
    """
    A mixin class that provides the steepest descent direction strategy,
    i.e., the Cauchy direction, which is the negative gradient direction.

    `p_k = -f'(x_k)`
    """

    def direction(self, x: floatVec, grad: floatVec) -> floatVec:
        return -grad


class ExactLineSearchMixin(LineSearchOptimiser):
    """
    A mixin class that provides the exact line search step length strategy for convex quadratic functions.

    `alpha_k = - (f'(x_k)^T p_k) / (p_k^T Q p_k)`\\
    where `Q` is the symmetric positive definite Hessian matrix of the convex quadratic function.
    """

    def initialise_state(self):
        super().initialise_state()

        Q = self.config.get("Q", None)
        if Q is None:
            raise ValueError("Q matrix is required for exact line search.")
        self.Q: floatVec = np.array(Q, dtype=float)

        self.alpha_thresh = float(self.config.get("alpha_thresh", 1e-14))
        self.alpha_min = float(self.config.get("alpha_min", 1e-8))

    def step_length(
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

        # Fallback if denominator is too small
        if abs(denom) < self.alpha_thresh:
            return self.alpha_min

        return -numer / denom


# ---------- Optimiser Implementations ----------
class SteepestGradientDescentExactLineSearch(
    SteepestDescentDirectionMixin, ExactLineSearchMixin, LineSearchOptimiser
):
    """
    Steepest gradient descent with exact line search for convex quadratic functions.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`\\
    where `alpha_k = (f'(x_k)^T f'(x_k)) / (f'(x_k)^T Q f'(x_k))`
    """

    pass  # All methods are provided by the mixins


class SteepestGradientDescentArmijo(SteepestDescentDirectionMixin, LineSearchOptimiser):
    """
    Forward-expansion Armijo line search:\\
    Increase alpha until Armijo condition holds (or until safe cap).

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k`
    """

    def initialise_state(self):
        super().initialise_state()

        self.c = float(self.config.get("c", 1e-4))  # Armijo parameter
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c < 1, "c must be in (0, 1)"

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        derphi0 = float(grad.T @ direction)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # Forward expansion
        alpha = self.alpha_init
        for _ in range(self.maxiter):
            f_new, _ = self._phi_and_derphi(x, alpha, direction, oracle_fn)
            if f_new <= f + self.c * alpha * derphi0:
                return alpha
            alpha *= self.beta
            if alpha < self.alpha_min or alpha > self.alpha_max:
                break
        return alpha


class SteepestGradientDescentArmijoGoldstein(
    SteepestDescentDirectionMixin, LineSearchOptimiser
):
    """
    Armijo-Goldstein via expansion to bracket and then bisection.\\
    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k` (Armijo)\\
    `f(x_k + alpha_k * p_k) >= f(x_k) + (1 - c) * alpha_k * f'(x_k)^T p_k` (Goldstein)
    """

    def initialise_state(self):
        super().initialise_state()

        self.c = float(self.config.get("c", 1e-4))  # Armijo-Goldstein parameter
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c < 0.5, "c must be in (0, 0.5)"

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        derphi0 = float(grad.T @ direction)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # If initial alpha already satisfies both, return it
        alpha = self.alpha_init
        f_new, _ = self._phi_and_derphi(x, alpha, direction, oracle_fn)
        if (f_new <= f + self.c * alpha * derphi0) and (
            f_new >= f + (1 - self.c) * alpha * derphi0
        ):
            return alpha

        # Expand to find an interval [alpha_lo, alpha_hi] where Armijo condition is satisfied at alpha_hi
        alpha_lo = 0.0
        alpha_hi = alpha
        for _ in range(self.maxiter):
            phi_hi, _ = self._phi_and_derphi(x, alpha_hi, direction, oracle_fn)
            if phi_hi <= f + self.c * alpha_hi * derphi0:
                break
            alpha_hi *= self.beta
            if alpha_hi > self.alpha_max:
                break

        # Bisect between alpha_lo and alpha_hi until Goldstein condition hold
        for _ in range(self.maxiter):
            alpha_mid = 0.5 * (alpha_lo + alpha_hi)
            phi_mid, _ = self._phi_and_derphi(x, alpha_mid, direction, oracle_fn)
            if (phi_mid <= f + self.c * alpha_mid * derphi0) and (
                phi_mid >= f + (1 - self.c) * alpha_mid * derphi0
            ):
                return alpha_mid
            # If phi_mid is too large -> need smaller step (move hi)
            if phi_mid > f + self.c * alpha_mid * derphi0:
                alpha_hi = alpha_mid
            else:
                # phi_mid < lower bound, it may violate upper bound -> move lo
                alpha_lo = alpha_mid
            if abs(alpha_hi - alpha_lo) < self.alpha_min:
                break
        return 0.5 * (alpha_lo + alpha_hi)


class SteepestGradientDescentWolfe(SteepestDescentDirectionMixin, LineSearchOptimiser):
    """
    Strong Wolfe line search using bracket + zoom (Nocedal & Wright).\\
    `phi(alpha_k) <= phi(0) + c1 * alpha_k * phi'(0)` (Armijo)\\
    `|phi'(alpha_k)| <= c2 * |phi'(0)|` (Strong curvature)\\
    where `phi(alpha_k) = f(x_k + alpha_k * p_k)`, `phi'(alpha_k) = f'(x_k + alpha_k * p_k)^T p_k`.
    """

    def initialise_state(self):
        super().initialise_state()

        self.c1 = float(self.config.get("c1", 1e-4))
        self.c2 = float(self.config.get("c2", 0.9))
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c1 < self.c2 < 1, "0 < c1 < c2 < 1 must be satisfied"

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        derphi0 = float(grad.T @ direction)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        phi0 = f
        alpha = self.alpha_init
        phi_prev = phi0
        alpha_prev = 0.0

        for i in range(self.maxiter):
            phi_a, derphi_a = self._phi_and_derphi(x, alpha, direction, oracle_fn)

            # Check Armijo
            if (phi_a > phi0 + self.c1 * alpha * derphi0) or (
                i > 0 and phi_a >= phi_prev
            ):
                # bracket found between alpha_prev and alpha
                return self._zoom(
                    oracle_fn, x, direction, alpha_prev, alpha, phi0, derphi0
                )
            # Check strong Wolfe
            if abs(derphi_a) <= self.c2 * abs(derphi0):
                return alpha
            # If derivative is positive, bracket and zoom
            if derphi_a >= 0:
                return self._zoom(
                    oracle_fn, x, direction, alpha, alpha_prev, phi0, derphi0
                )
            # Otherwise increase alpha (extrapolate)
            alpha_prev = alpha
            phi_prev = phi_a
            alpha = alpha * self.beta
        return alpha  # Fallback

    def _zoom(
        self,
        oracle_fn: FirstOrderOracle,
        x: floatVec,
        d: floatVec,
        alpha_lo: float,
        alpha_hi: float,
        phi0: float,
        derphi0: float,
        maxiter: int = 50,
    ):
        """
        Zoom procedure as in Nocedal & Wright (uses safe bisection interpolation).
        Returns an alpha that satisfies strong Wolfe (if found), otherwise the best found.
        """
        phi_lo, _derphi_lo = self._phi_and_derphi(x, alpha_lo, d, oracle_fn)
        for _ in range(maxiter):
            alpha_j = 0.5 * (alpha_lo + alpha_hi)  # safe midpoint
            phi_j, derphi_j = self._phi_and_derphi(x, alpha_j, d, oracle_fn)

            # Armijo condition
            if (phi_j > phi0 + self.c1 * alpha_j * derphi0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j
            else:
                # Check strong Wolfe condition (absolute derivative)
                if abs(derphi_j) <= self.c2 * abs(derphi0):
                    return alpha_j
                # If derivative has same sign as _derphi_lo, shrink interval
                if derphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
                phi_lo = phi_j
                _derphi_lo = derphi_j
            if abs(alpha_hi - alpha_lo) < 1e-14:
                break
        # fallback: return midpoint
        return 0.5 * (alpha_lo + alpha_hi)


class SteepestGradientDescentBacktracking(
    SteepestDescentDirectionMixin, LineSearchOptimiser
):
    """
    Standard backtracking Armijo (decreasing alpha).
    """

    def initialise_state(self):
        super().initialise_state()

        self.c = float(self.config.get("c", 1e-4))  # Armijo parameter
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c < 1, "c must be in (0, 1)"

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        direction: floatVec,
        oracle_fn: FirstOrderOracle,
    ) -> float:
        derphi0 = float(grad.T @ direction)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        alpha = self.alpha_init
        for _ in range(self.maxiter):
            new_f, _ = self._phi_and_derphi(x, alpha, direction, oracle_fn)
            if new_f <= f + self.c * alpha * derphi0:
                return alpha
            alpha *= self.beta
            if alpha < self.alpha_min:
                return self.alpha_min
        return alpha


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
        """Solves the linear system `Ax = b`."""
        if self.m == self.n:
            self._solution_method = "direct"
            return np.linalg.solve(self.A, self.b)
        else:
            self._solution_method = "least_squares"
            x, *_ = np.linalg.lstsq(self.A, self.b, rcond=None)
            return x

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

    @property
    def solution_method(self) -> str:
        """Returns the method used to solve the system."""
        return getattr(self, "_solution_method", "unsolved")


def format_float(
    obj: float | floatVec,
    dprec: int = 2,
    fprec: int = 16,
    ffmt: str = "f",
    sep: str = ",\n",
    lim: int = 5,
) -> str | None:
    _fmt = f"{{:{dprec}.{fprec}{ffmt}}}"
    if isinstance(obj, float):
        return f"{_fmt.format(obj)}"
    elif isinstance(obj, np.ndarray):
        if obj.size <= lim:
            formatted = [f"{_fmt.format(x)}" for x in obj]
        else:
            # Show first two and last two items
            formatted = (
                [f"{_fmt.format(x)}" for x in obj[:2]]
                + ["..."]
                + [f"{_fmt.format(x)}" for x in obj[-2:]]
            )
        return "[" + sep.join(formatted) + "]"


def format_time(t: float | None) -> str:
    """Format time in seconds to an appropriate unit"""
    if t is None:
        return "N/A"
    abs_t = abs(t)
    if abs_t >= 60:
        minutes = int(t // 60)
        seconds = t % 60
        if seconds < 1e-3:
            return f"{minutes} min"
        return f"{minutes} min {round(seconds)} s"
    units: list[tuple[str, float]] = [("s", 1), ("ms", 1e-3), ("\u03bcs", 1e-6)]
    for unit, thresh in units:
        if abs_t >= thresh:
            val = t / thresh
            if unit == "\u03bcs":
                return f"{int(round(val))} {unit}"
            if unit == "min":
                return f"{val:.2f} {unit}"
            return f"{val:.3f} {unit}"
    return f"{int(round(t / 1e-6))} \u03bcs"  # Fallback for very small values


# ---------- Questions ----------
def question_1():
    console.rule("[bold magenta]Question 1", style="magenta")

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

        x0s = [np.array([0.0, 0.0])]
        # Any starting point should work, since a local minima for the
        # convex quadratic function with Q >> 0 will be the global minima

        optim.run(oracle, x0s=x0s, maxiter=10_000, tol=1e-13, show_params=False)

        x_star_analytical = -np.linalg.solve(Q, b)
        fx_star_analytical, dfx_star_analytical = oracle(x_star_analytical)
        console.print("\n[bold green]Analytical solution:[/]")
        print(f"x* = {format_float(x_star_analytical, sep=', ')}")
        print(f"f(x*) = {format_float(fx_star_analytical)}")
        print(f"f'(x*) = {format_float(dfx_star_analytical, sep=', ')}")

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
    console.rule("[bold magenta]Question 2", style="magenta")

    def grad_fn(srn: int, x: floatVec) -> floatVec:
        g: floatVec = oq2g(srn, x)
        return g.reshape(-1)

    oracle = FirstOrderOracle.from_separate(oq2f, grad_fn, dim=5)

    optimisers: list[LineSearchOptimiser] = [
        SteepestGradientDescentArmijo(alpha=0.3, beta=0.8, initial_step_length=1.0),
        SteepestGradientDescentArmijoGoldstein(
            alpha=0.3, beta=0.8, initial_step_length=1.0
        ),
        SteepestGradientDescentWolfe(alpha=0.3, beta=0.8, initial_step_length=1.0),
        SteepestGradientDescentBacktracking(
            alpha=0.3, beta=0.8, initial_step_length=1.0
        ),
    ]
    x0s = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, -1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
    ]
    plt.figure()
    for i, optim in enumerate(optimisers):
        if i != 0:
            console.rule(style="default")
        optim.run(oracle, x0s=x0s, maxiter=1_000, tol=1e-7)
        optim.plot_step_lengths()
    plt.title(r"Step length vs Iteration for different (Inexact) Line Search methods")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Step length $\alpha_k$")
    plt.legend()
    plt.grid(True)


def question_3():
    console.rule("[bold magenta]Question 3", style="magenta")

    question_3_2()
    console.rule(style="default")
    question_3_5()


def question_3_2():
    t0 = time.perf_counter()
    with console.status("Loading A, b from oracle..."):
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

    optim = SteepestGradientDescentWolfe(
        alpha=0.01, beta=0.9, initial_step_size=1.0, maxiter=10
    )
    x0s = [np.zeros(ls.n)]
    optim.run(oracle, x0s=x0s, maxiter=10, tol=1e-3)


def question_3_5():
    gen_times: list[float | None] = []
    inv_times: list[float | None] = []
    opt_times: list[float | None] = []
    npy_times: list[float | None] = []

    m_values: list[int] = [2**i for i in range(1, 13)]
    for m in m_values:
        try:
            # Generate a random well-conditioned A, cond(A) ~ 1 + 1e3
            console.print(f"[cyan]LinearSystem [italic]with {m=}[/]")
            t0 = time.perf_counter()
            with console.status("Generating A and b..."):
                A: floatVec = np.eye(m) + 1e-3 * np.random.randn(m, m)
                b: floatVec = np.random.randn(m)
            t1 = time.perf_counter()
            t = t1 - t0
            gen_times.append(t)
            console.print(
                f"[bright_black]Time taken to generate A, b: {t:.6f} seconds[/]"
            )
            ls = LinearSystem(A, b)

            t0 = time.perf_counter()
            with console.status("Solving system using matrix inversion..."):
                _x = -np.linalg.inv(A) @ b
            t1 = time.perf_counter()
            t = t1 - t0
            inv_times.append(t)
            console.print(
                f"[bright_black]Time taken to solve system using matrix inversion: {t:.6f} seconds[/]"
            )

            oracle = ls.make_oracle()
            optim = SteepestGradientDescentWolfe(
                alpha=0.01, beta=0.9, initial_step_length=1.0
            )
            x0s = [np.zeros(ls.n)]
            t0 = time.perf_counter()
            optim.run(oracle, x0s=x0s, maxiter=1_000, tol=1e-6, show_params=False)
            t1 = time.perf_counter()
            t = t1 - t0
            opt_times.append(t)
            console.print(
                f"[bright_black]Time taken to solve system using optimisation techniques: {t:.6f} seconds[/]"
            )

            t0 = time.perf_counter()
            with console.status("Solving system using numpy.linalg.solve..."):
                ls.solve()
            t1 = time.perf_counter()
            t = t1 - t0
            npy_times.append(t)
            console.print(
                f"[bright_black]Time taken to solve system using numpy.linalg.solve: {t:.6f} seconds[/]"
            )

            print()
        except MemoryError as e:
            console.print(
                f"[yellow][bold]MemoryError:[/bold] Unable to allocate [italic]for {m=}:[/italic][/yellow] {e}"
            )
            gen_times.append(None)
            inv_times.append(None)
            opt_times.append(None)
            npy_times.append(None)
        except Exception as e:
            console.print(
                f"[red][bold]Error[/bold] [italic]for {m=}:[/italic][/red] {e}"
            )
            gen_times.append(None)
            inv_times.append(None)
            opt_times.append(None)
            npy_times.append(None)

    table = Table(title="Timing results for solving linear systems")
    table.add_column("Dimensions (m\u00d7m)", justify="center")
    table.add_column("Generation", justify="center")
    table.add_column("Matrix inversion", justify="center")
    table.add_column("Optimisation", justify="center")
    table.add_column("numpy.linalg.solve", justify="center")
    for i, m in enumerate(m_values):
        table.add_row(
            f"{m}\u00d7{m}",
            format_time(gen_times[i]),
            format_time(inv_times[i]),
            format_time(opt_times[i]),
            format_time(npy_times[i]),
        )
    console.print(table)


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()

    plt.show()
