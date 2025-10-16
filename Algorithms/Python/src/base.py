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
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import TextType

from .functions import ConvexQuadratic
from .oracle import FirstOrderOracle
from .types import floatVec
from .utils import format_float, format_time, show_solution

console = Console()


# ---------- Algorithm Templates ----------
class IterativeOptimiser(ABC):
    """
    A base template class for iterative optimisation algorithms,
    particularly for the minimisation objective.

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

    @abstractmethod
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
        log_runs: bool = False,
    ):
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0s: Initial guesses for the minimum point.
            maxiter: Maximum number of iterations to perform.
            tol: Tolerance for stopping criterion based on the gradient.
            show_params: Whether to display the configuration parameters of the algorithm.
            log_runs: Log all the runs over different initial points, not just the best one.
        """
        self.maxiter = maxiter
        self.tol = tol

        console = Console()
        console.print(f"[bold blue]{self.name}[/]")
        if show_params and self.config:
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
            if log_runs and has_multiple_x0:
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


class LineSearchOptimiser(IterativeOptimiser, ABC):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `alpha_k` is the step length along the descent direction `p_k`.
    """

    def initialise_state(self):
        super().initialise_state()
        self.step_lengths: list[float] = []
        self.step_directions: list[floatVec] = []

    def direction(
        self, x: floatVec, k: int, f: float, grad: floatVec, oracle_fn: FirstOrderOracle
    ) -> floatVec:
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
        oracle_fn: FirstOrderOracle,
        direction: floatVec,
    ) -> float:
        """
        Returns step length `alpha_k` to take along the descent direction `p_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step length strategy.
        """
        raise NotImplementedError

    def step(self, x, k, f, grad, oracle_fn):
        p_k = self.direction(x, k, f, grad, oracle_fn)
        self.step_directions.append(p_k)

        alpha_k = self.step_length(x, k, f, grad, oracle_fn, p_k)
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

    def direction(
        self, x: floatVec, k: int, f: float, grad: floatVec, oracle_fn: FirstOrderOracle
    ) -> floatVec:
        return -grad


class ExactLineSearchMixin(LineSearchOptimiser):
    """
    A mixin class that provides the exact line search step length strategy for convex quadratic functions.

    `alpha_k = - (f'(x_k)^T p_k) / (p_k^T Q p_k)`\\
    where `Q` is the symmetric positive definite Hessian matrix of the convex quadratic function.
    """

    def initialise_state(self):
        super().initialise_state()

    def step_length(
        self,
        x: floatVec,
        k: int,
        f: float,
        grad: floatVec,
        oracle_fn: FirstOrderOracle,
        direction: floatVec,
    ) -> float:
        if not isinstance(oracle_fn._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        Q = oracle_fn._oracle_f.Q
        numer = float(grad.T @ direction)
        denom = float(direction.T @ Q @ direction)
        alpha = numer / denom

        if alpha < 0:
            alpha = -alpha
            self.step_directions[-1] = -direction

        return alpha
