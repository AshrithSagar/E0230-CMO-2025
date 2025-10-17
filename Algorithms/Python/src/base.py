"""
src/base.py
=======
Algorithm base and mixin classes.

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import TextType

from .functions import ConvexQuadratic
from .oracle import AbstractOracle, FirstOrderOracle
from .stopping import (
    CompositeCriterion,
    GradientNormCriterion,
    MaxIterationsCriterion,
    StoppingCriterion,
)
from .types import floatVec
from .utils import format_float, format_time, show_solution

console = Console()


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

    def reset(self) -> None:
        """
        Resets the internal state of the algorithm before a new run.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    @abstractmethod
    def step(self, x: floatVec, k: int, oracle_fn: AbstractOracle) -> floatVec:
        """
        Performs a single step of the algorithm.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            x: Current value of `x`, i.e., `x_k`.
            k: Current iteration number.
            oracle_fn: The oracle function to query for `f(x)`.
        Returns:
            The updated value of `x` after the step, viz. `x_{k+1}`.
        """
        raise NotImplementedError

    def run(
        self,
        oracle_fn: AbstractOracle,
        x0: floatVec,
        criteria: Optional[
            Union[StoppingCriterion, CompositeCriterion, Iterable[StoppingCriterion]]
        ] = None,
        show_params: bool = True,
    ) -> dict[str, Any]:
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0: Initial guess for the minimum point.
            criteria: List of stopping criteria to determine when to stop the optimisation.
            show_params: Whether to display the configuration parameters of the algorithm.

        Returns:
            A dictionary containing the results of the optimisation run, including:
            - 'x0': Initial point.
            - 'x_star': Estimated minimum point.
            - 'f_star': Function value at the estimated minimum point.
            - 'n_iters': Number of iterations performed.
            - 'history': List of `x` values at each iteration.
            - 'oracle_call_count': Total number of oracle calls made.
            - 'time_taken': Total time taken for the optimisation run.
        """

        console = Console()
        console.print(f"[bold blue]{self.name}[/]")
        if show_params and self.config:
            console.print(f"params: {self.config}")

        _crit: list[StoppingCriterion] = []
        if criteria is None:
            criteria = _crit
            if isinstance(oracle_fn, FirstOrderOracle):
                _crit.append(GradientNormCriterion(tol=1e-6))
        elif isinstance(criteria, CompositeCriterion):
            for crit in criteria.criteria:
                _crit.append(crit)
        elif isinstance(criteria, StoppingCriterion):
            _crit.append(criteria)
        elif isinstance(criteria, Iterable):
            for crit in criteria:
                _crit.append(crit)
        maxiter: float = float("inf")
        for crit in _crit:
            if isinstance(crit, MaxIterationsCriterion):
                maxiter = min(maxiter, crit.maxiter)
        if maxiter == float("inf"):
            maxiter = 1000
            _crit.append(MaxIterationsCriterion(maxiter))
        maxiter = int(maxiter)
        criteria = CompositeCriterion(_crit)

        k: int = 0
        x: floatVec = x0
        history: list[floatVec] = [x0]
        self.reset()
        oracle_fn.reset()
        criteria.reset()

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("iter:{task.completed:04},"),
            TextColumn("Oracle calls: {task.fields[oracle_calls]:04}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        t0 = time.perf_counter()
        try:
            task = progress.add_task("Running...", total=maxiter, oracle_calls=0)
            for k in range(maxiter):
                progress.update(task, advance=1, oracle_calls=oracle_fn.call_count)
                if criteria.check(x, k, oracle_fn):
                    break
                x = self.step(x, k, oracle_fn)
                history.append(x)
        except OverflowError:  # Fallback, in case of non-convergence
            x = np.full(oracle_fn.dim, np.nan)
        finally:
            progress.stop()
        t1 = time.perf_counter()
        t = t1 - t0

        fx = oracle_fn._oracle_f.eval(x)
        n_iters = k + 1
        n_oracle = oracle_fn.call_count
        info = {
            "x0": x0,
            "x_star": x,
            "f_star": fx,
            "n_iters": n_iters,
            "history": history,
            "oracle_call_count": n_oracle,
            "time_taken": t,
        }
        self._show_run_result(x, fx, x0, n_iters, n_oracle)
        console.print(f"[bright_black]Time taken: {format_time(t)}[/]")
        return info

    def _show_run_result(
        self,
        x: floatVec,
        fx: float,
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
        show_solution(x, fx, table=table)


class LineSearchOptimiser(IterativeOptimiser):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `alpha_k` is the step length along the descent direction `p_k`.
    """

    def reset(self):
        super().reset()
        self.step_lengths: list[float] = []
        self.step_directions: list[floatVec] = []

    def direction(self, x: floatVec, k: int, oracle_fn: FirstOrderOracle) -> floatVec:
        """
        Returns the descent direction `p_k` to move towards from `x_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific direction strategy.
        """
        raise NotImplementedError

    def step_length(
        self, x: floatVec, k: int, oracle_fn: FirstOrderOracle, direction: floatVec
    ) -> float:
        """
        Returns step length `alpha_k` to take along the descent direction `p_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step length strategy.
        """
        raise NotImplementedError

    def step(self, x, k, oracle_fn):
        assert isinstance(oracle_fn, FirstOrderOracle), (
            f"{self.__class__.__name__} requires a FirstOrderOracle."
        )

        p_k = self.direction(x, k, oracle_fn)
        self.step_directions.append(p_k)

        alpha_k = self.step_length(x, k, oracle_fn, p_k)
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

    def direction(self, x: floatVec, k: int, oracle_fn: FirstOrderOracle) -> floatVec:
        fx, dfx = oracle_fn(x)
        return -dfx


class ExactLineSearchMixin(LineSearchOptimiser):
    """
    A mixin class that provides the exact line search step length strategy for convex quadratic functions.

    `alpha_k = - (f'(x_k)^T p_k) / (p_k^T Q p_k)`\\
    where `Q` is the symmetric positive definite Hessian matrix of the convex quadratic function.
    """

    def step_length(
        self, x: floatVec, k: int, oracle_fn: FirstOrderOracle, direction: floatVec
    ) -> float:
        if not isinstance(oracle_fn._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        Q = oracle_fn._oracle_f.Q
        _, grad = oracle_fn(x)
        numer = float(grad.T @ direction)
        denom = float(direction.T @ Q @ direction)
        alpha = numer / denom

        if alpha < 0:
            alpha = -alpha
            self.step_directions[-1] = -direction

        return alpha
