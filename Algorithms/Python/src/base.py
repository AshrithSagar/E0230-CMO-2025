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
from typing import Any, Generic, Iterable, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import TextType

from .functions import ConvexQuadratic
from .oracle import AbstractOracle, FirstOrderOracle, SecondOrderOracle, ZeroOrderOracle
from .stopping import (
    CompositeCriterion,
    GradientNormCriterion,
    MaxIterationsCriterion,
    StoppingCriterion,
)
from .types import floatMat, floatVec
from .utils import format_float, format_time, show_solution

console = Console()

TStepInfo = TypeVar("TStepInfo", bound="StepInfo")
"""Generic type for Step classes."""


class StepInfo:
    """
    A base class for Step information.
    Encapsulates the state of the algorithm at iteration `k`.
    """

    def __init__(self, x: floatVec, k: int, oracle: AbstractOracle):
        self.x: floatVec = x
        """The current point `x_k` in the input space."""

        self.k: int = k
        """The current iteration number `k`."""

        self.oracle: AbstractOracle = oracle
        """The oracle function used to evaluate `f`."""

        # Internal values
        self._fx: Optional[float] = None
        self._dfx: Optional[floatVec] = None
        self._d2fx: Optional[floatMat] = None

    @property
    def fx(self) -> float:
        """The function value `f(x_k)` at the current point."""
        if self._fx is None:
            if isinstance(self.oracle, ZeroOrderOracle):
                (f,) = self.oracle(self.x)
                self._fx = f
            elif isinstance(self.oracle, FirstOrderOracle):
                f, g = self.oracle(self.x)
                self._fx = f
                self._dfx = g
            elif isinstance(self.oracle, SecondOrderOracle):
                f, g, h = self.oracle(self.x)
                self._fx = f
                self._dfx = g
                self._d2fx = h
            else:
                raise RuntimeError("Function value not available with this oracle.")
        return self._fx

    @property
    def grad(self) -> floatVec:
        """The gradient `f'(x_k)` at the current point."""
        if self._dfx is None:
            if isinstance(self.oracle, FirstOrderOracle):
                f, g = self.oracle(self.x)
                self._fx = f
                self._dfx = g
            elif isinstance(self.oracle, SecondOrderOracle):
                f, g, h = self.oracle(self.x)
                self._fx = f
                self._dfx = g
                self._d2fx = h
            else:
                raise RuntimeError("Gradient not available with this oracle.")
        return self._dfx

    @property
    def hess(self) -> floatMat:
        """The Hessian `f''(x_k)` at the current point."""
        if self._d2fx is None:
            if isinstance(self.oracle, SecondOrderOracle):
                f, g, h = self.oracle(self.x)
                self._fx = f
                self._dfx = g
                self._d2fx = h
            else:
                raise RuntimeError("Hessian not available with this oracle.")
        return self._d2fx


class IterativeOptimiser(ABC, Generic[TStepInfo]):
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

    def _make_step_info(
        self, x: floatVec, k: int, oracle_fn: AbstractOracle
    ) -> TStepInfo:
        """
        Helper to create StepInfo instances.
        Can be overridden by subclasses to provide custom StepInfo types.
        """
        return StepInfo(x, k, oracle_fn)  # type: ignore

    def reset(self) -> None:
        """
        Resets the internal state of the algorithm before a new run.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    @abstractmethod
    def step(self, info: TStepInfo) -> floatVec:
        """
        Performs a single step of the algorithm.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            info: An instance of `StepInfo` containing the current state of the algorithm.
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
                # Default criterion for first-order methods, if unspecified
                _crit.append(GradientNormCriterion(tol=1e-6))
        elif isinstance(criteria, CompositeCriterion):
            _crit.extend(criteria.criteria)
        elif isinstance(criteria, StoppingCriterion):
            _crit.append(criteria)
        elif isinstance(criteria, Iterable):
            _crit.extend(criteria)
        maxiter: float = float("inf")
        for crit in _crit:
            if isinstance(crit, MaxIterationsCriterion):
                maxiter = min(maxiter, crit.maxiter)
        if maxiter == float("inf"):
            maxiter = 1000  # Default maxiter, if unspecified
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
                info = self._make_step_info(x, k, oracle_fn)
                x = self.step(info)
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


class LineSearchStepInfo(StepInfo):
    def __init__(self, x: floatVec, k: int, oracle: FirstOrderOracle):
        super().__init__(x, k, oracle)
        self.direction: Optional[floatVec] = None
        self.alpha: Optional[float] = None


class LineSearchOptimiser(IterativeOptimiser[LineSearchStepInfo]):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `alpha_k` is the step length along the descent direction `p_k`.
    """

    def _make_step_info(
        self, x: floatVec, k: int, oracle_fn: AbstractOracle
    ) -> LineSearchStepInfo:
        assert isinstance(oracle_fn, FirstOrderOracle)
        return LineSearchStepInfo(x, k, oracle_fn)

    def reset(self):
        super().reset()
        self.step_lengths: list[float] = []
        self.step_directions: list[floatVec] = []

    def direction(self, info: LineSearchStepInfo) -> floatVec:
        """
        Returns the descent direction `p_k` to move towards from `x_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific direction strategy.
        """
        raise NotImplementedError

    def step_length(self, info: LineSearchStepInfo) -> float:
        """
        Returns step length `alpha_k` to take along the descent direction `p_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step length strategy.
        """
        raise NotImplementedError

    def step(self, info: LineSearchStepInfo) -> floatVec:
        p_k = self.direction(info)
        self.step_directions.append(p_k)

        alpha_k = self.step_length(info)
        self.step_lengths.append(alpha_k)

        return info.x + alpha_k * p_k

    def plot_step_lengths(self):
        """Plot step lengths vs iterations for the best run."""
        plt.plot(self.step_lengths, marker="o", label=self.name)

    def _phi_and_derphi(self, info: LineSearchStepInfo, alpha: float):
        """Computes\\
        `phi(alpha) = f(x + alpha * d)`\\
        `phi'(alpha) = f'(x + alpha * d)^T d`
        """
        if info.direction is None:
            raise ValueError("Direction not set in StepInfo.")
        x, d = info.x, info.direction
        xa = x + alpha * d
        fa, g_a = info.oracle(xa)
        return fa, float(g_a.T @ d)


class SteepestDescentDirectionMixin(LineSearchOptimiser):
    """
    A mixin class that provides the steepest descent direction strategy,
    i.e., the Cauchy direction, which is the negative gradient direction.

    `p_k = -f'(x_k)`
    """

    def direction(self, info: LineSearchStepInfo) -> floatVec:
        return -info.grad


class ExactLineSearchMixin(LineSearchOptimiser):
    """
    A mixin class that provides the exact line search step length strategy for convex quadratic functions.

    `alpha_k = - (f'(x_k)^T p_k) / (p_k^T Q p_k)`\\
    where `Q` is the symmetric positive definite Hessian matrix of the convex quadratic function.
    """

    def step_length(self, info: LineSearchStepInfo) -> float:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )
        elif not isinstance(info.oracle, FirstOrderOracle):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a FirstOrderOracle."
            )
        elif info.direction is None:
            raise ValueError("Direction not set in StepInfo.")

        x, d = info.x, info.direction
        Q = info.oracle._oracle_f.Q
        _, grad = info.oracle(x)
        numer = float(grad.T @ d)
        denom = float(d.T @ Q @ d)
        alpha = numer / denom

        if alpha < 0:
            alpha = -alpha
            self.step_directions[-1] = -d

        return alpha
