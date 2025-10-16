"""
src/base.py
=======
Base templates for algorithms

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import TextType

from .oracle import ConvexQuadraticOracle, FirstOrderOracle
from .types import floatVec
from .utils import format_float, format_time, show_solution

console = Console()


LOG_RUNS: bool = False
"""Log all the runs of the optimisation algorithms in the summary table, not just the best one."""


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
        maxiter: int = 1_000,
        tol: float = 1e-6,
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

        console = Console()
        console.print(f"[bold blue]{self.name}[/]")
        if show_params:
            console.print(f"params: {self.config}")
        has_multiple_x0 = len(x0s) > 1

        self.runs = []
        t00 = time.perf_counter()
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
            t0 = time.perf_counter()
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
            t1 = time.perf_counter()
            t = t1 - t0

            self.runs.append(
                {
                    "x0": x0,
                    "x_star": x,
                    "fx_star": fx,
                    "dfx_star": dfx,
                    "history": history,
                    "oracle_call_count": oracle_fn.call_count,
                    "time_taken": t,
                }
            )
            if LOG_RUNS and has_multiple_x0:
                title = f"[not italic][bold yellow]Run {idx}:[/]"
                self._show_run_result(
                    x, fx, dfx, x0, len(history) - 1, oracle_fn.call_count, title
                )
                console.print(f"[bright_black]Time taken: {format_time(t)}[/]")
        t10 = time.perf_counter()

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

        title = (
            f"[not italic][bold green]Best run (Run {run_idx}):[/]"
            if has_multiple_x0
            else ""
        )
        self._show_run_result(
            self.x_star, self.fx_star, self.dfx_star, x0, n_iters, n_oracle, title
        )
        console.print(f"[bright_black]Time taken: {format_time(t10 - t00)}[/]")

    def plot_history(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)

    def _show_run_result(
        self,
        x: floatVec,
        fx: float,
        dfx: floatVec,
        x0: floatVec,
        n_iters: int | str,
        n_oracle: int | str,
        title: TextType | None = None,
    ) -> None:
        """Helper to format and display the values"""
        table = Table(title=title, title_justify="left", show_header=False)
        table.add_column(style="bold", justify="right")
        table.add_column()
        table.add_row("x0", format_float(x0, sep=", "))
        table.add_row("Iterations", str(n_iters))
        table.add_row("Oracle calls", str(n_oracle))
        table.add_section()
        show_solution(x, fx, dfx, table=table)


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

        self.denom_thresh = float(self.config.get("denom_thresh", 1e-14))
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
        if not isinstance(oracle_fn, ConvexQuadraticOracle):
            raise NotImplementedError(
                "This implementation of exact line search requires a ConvexQuadraticOracle."
            )

        numer = float(grad.T @ direction)
        denom = float(direction.T @ self.Q @ direction)

        # Fallback if denominator is too small
        if abs(denom) < self.denom_thresh:
            return self.alpha_min

        return -numer / denom


# ---------- Optimiser Implementations ----------
class SteepestGradientDescent(SteepestDescentDirectionMixin, IterativeOptimiser):
    """
    Steepest gradient descent.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`
    """

    def step(self, x, k, f, grad, oracle_fn):
        p_k = self.direction(x, grad)
        alpha_k = 1e-3
        return x + alpha_k * p_k


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
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_start = float(self.config.get("alpha_start", 0.0))
        self.alpha_step = float(self.config.get("alpha_step", 1e-1))
        self.alpha_stop = float(self.config.get("alpha_stop", 1.0))

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
        alpha_prev: float | None = None
        for alpha in np.arange(
            self.alpha_start, self.alpha_stop + self.alpha_step, self.alpha_step
        ):
            f_new, _ = self._phi_and_derphi(x, float(alpha), direction, oracle_fn)
            if f_new <= f + self.c * alpha * derphi0:
                alpha_prev = float(alpha)
            else:
                break
        if alpha_prev is not None:
            return alpha_prev
        else:
            return self.alpha_min


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


class ConjugateDirectionMethod(IterativeOptimiser):
    """
    Linear conjugate direction method for convex quadratic functions.\\
    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    def initialise_state(self):
        super().initialise_state()
        self.Q: floatVec = np.array(self.config.get("Q"), dtype=float)
        if self.Q is None:
            raise ValueError("Q matrix is required for conjugate direction method.")
        self.denom_thresh = float(self.config.get("denom_thresh", 1e-14))
        self.alpha_min = float(self.config.get("alpha_min", 1e-8))

        self.r: floatVec | None = None  # Residual (negative gradient)
        self.p: floatVec | None = None  # Search direction
        self.k_prev: int | None = None  # Previous iteration number

    def step(self, x, k, f, grad, oracle_fn):
        r_k = -grad  # Residual is negative gradient
        if self.r is None or self.p is None or self.k_prev is None or k == 1:
            # First iteration or reset
            p_k = r_k
        else:
            beta_k = float((r_k.T @ r_k) / (self.r.T @ self.r))
            p_k = r_k + beta_k * self.p

        numer = float(r_k.T @ p_k)
        denom = float(p_k.T @ self.Q @ p_k)

        # Fallback if denominator is too small
        if abs(denom) < self.denom_thresh:
            alpha_k = self.alpha_min
        else:
            alpha_k = -numer / denom

        # Update state
        self.r = r_k
        self.p = p_k
        self.k_prev = k

        return x + alpha_k * p_k


class ConjugateGradientMethod(ConjugateDirectionMethod):
    """
    Linear conjugate gradient method for convex quadratic functions.\\
    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    def step(self, x, k, f, grad, oracle_fn):
        r_k = -grad  # Residual is negative gradient
        if self.r is None or self.p is None or self.k_prev is None or k == 1:
            # First iteration or reset
            p_k = r_k
        else:
            beta_k = float((r_k.T @ r_k) / (self.r.T @ self.r))
            p_k = r_k + beta_k * self.p

        numer = float(r_k.T @ p_k)
        denom = float(p_k.T @ self.Q @ p_k)

        # Fallback if denominator is too small
        if abs(denom) < self.denom_thresh:
            alpha_k = self.alpha_min
        else:
            alpha_k = -numer / denom

        # Update state
        self.r = r_k
        self.p = p_k
        self.k_prev = k

        return x + alpha_k * p_k
