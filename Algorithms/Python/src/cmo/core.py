"""
Algorithm base and mixin classes.
=======
src/cmo/core.py

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import TextType
from rich.tree import Tree

from .functions import ConvexQuadratic
from .info import (
    FirstOrderLineSearchStepInfo,
    FirstOrderStepInfo,
    LineSearchStepInfo,
    QuasiNewtonStepInfo,
    RunInfo,
    SecondOrderLineSearchStepInfo,
    StepInfo,
    ZeroOrderLineSearchStepInfo,
    ZeroOrderStepInfo,
)
from .logging import logger
from .oracle import FirstOrderOracle, Oracle, SecondOrderOracle, ZeroOrderOracle
from .stopping import (
    CompositeCriterion,
    MaxIterationsCriterion,
    StoppingCriterion,
    StoppingCriterionType,
)
from .types import Matrix, Scalar, Vector
from .utils import format_subscript, format_time, format_value, show_solution


class IterativeOptimiser[O: Oracle, T: StepInfo[Any]](ABC):
    """
    A base template class for iterative optimisation algorithms,
    particularly for the minimisation objective.

    `x_{k+1} = ALGO(x_k)`\\
    where `ALGO` is the algorithm-specific step function,
    `x_k` is the value at iteration `k`.
    """

    StepInfoClass: type[T]

    def __init__(self, **kwargs: Any) -> None:
        # Initialises the iterative optimiser with configuration parameters.
        self.config = kwargs

        self.name = self.__class__.__name__
        """Name of the algorithm, derived from the class name of the optimiser."""

    def reset(self) -> Self:
        """
        Resets the internal state of the algorithm before a new run.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        logger.debug(f"Optimiser [yellow]{self.name}[/] state has been reset.")
        return self

    @abstractmethod
    def step(self, info: T) -> T:
        """
        Performs a single step of the algorithm.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            info: An instance of `StepInfo` containing the current state of the algorithm at iteration `k`.
        Returns:
            An instance of `StepInfo` containing the updated state after the step for iteration `k+1`.
        """
        raise NotImplementedError

    @property
    def stopping(self) -> list[StoppingCriterionType[T]]:
        """Any stopping criteria defined by the optimiser itself."""
        return []

    def run(
        self,
        oracle_fn: O,
        x0: Vector,
        criteria: Optional[StoppingCriterionType[T]] = None,
        show_params: bool = True,
    ) -> RunInfo[O, T]:
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0: Initial guess for the minimum point.
            criteria: List of stopping criteria to determine when to stop the optimisation.
            show_params: Whether to display the configuration parameters of the algorithm.

        Returns:
            An instance of `RunInfo` containing details about the optimisation run.
        """

        console = Console()
        console.print(f"[bold blue]{self.name}[/]")
        if show_params and self.config:
            console.print(f"params: {self.config}")

        logger.info(f"Optimiser: [bold violet]{self.name}[/]")
        logger.info(f"Oracle: [bold violet]{oracle_fn.__class__.__name__}[/]")
        logger.info(
            f"Initial point [bold magenta]\U0001d431\u2080[/] = {format_value(x0, sep=', ')}"
        )
        logger.info(f"Config parameters: {self.config}")

        _crit: list[StoppingCriterion[Any]] = []
        for crit in [self.stopping, criteria]:
            if crit is None:
                continue
            elif isinstance(crit, CompositeCriterion):
                _crit.extend(crit.criteria)
            elif isinstance(crit, StoppingCriterion):
                _crit.append(crit)
            elif isinstance(crit, Iterable):
                _crit.extend(crit)
        maxiter = Scalar("inf")
        for crit in _crit:
            if isinstance(crit, MaxIterationsCriterion):
                maxiter = min(maxiter, crit.maxiter)
        if maxiter == Scalar("inf"):
            maxiter = 1000  # Default maxiter, if unspecified
            _crit.append(MaxIterationsCriterion(maxiter))
        maxiter = int(maxiter)
        criteria = CompositeCriterion(_crit)

        tree = Tree("Stopping criteria")
        for crit in criteria.criteria:
            tree.add(f"{crit}")
        with console.capture() as capture:
            console.print(tree)
        tree_str = capture.get()
        logger.info(tree_str.strip())

        self.reset()
        oracle_fn.reset()
        criteria.reset()

        k: int = 0
        x: Vector = x0
        info = self.StepInfoClass(k, x, oracle_fn)
        history: list[T] = [info]

        progress = None
        if not logger.isEnabledFor(logging.DEBUG):
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
            task = None
            if progress is not None:
                task = progress.add_task("Running...", total=maxiter, oracle_calls=0)

            for k in range(maxiter):
                logger.debug(f"[bold underline]Iteration {k}[/]:")
                if progress is not None and task is not None:
                    progress.update(task, advance=1, oracle_calls=oracle_fn.call_count)

                if criteria.check(info):
                    break
                info_next = self.step(info)
                if criteria.check(info):
                    break

                history.append(info_next)
                info = info_next

            x = info.x

        except OverflowError:  # Fallback, in case of non-convergence
            x = Vector(np.full(oracle_fn.dim, np.nan))

        finally:
            if progress is not None:
                progress.stop()

        t1 = time.perf_counter()
        t = t1 - t0

        logger.info(f"Optimisation finished in {k + 1} iterations.")
        fx = oracle_fn._oracle_f.eval(x)
        n_iters = k + 1
        n_oracle = oracle_fn.call_count
        self._show_run_result(x, fx, x0, n_iters, n_oracle)
        console.print(f"[dim]Time taken: [bold default]{format_time(t)}[/]")
        return RunInfo(
            x0=x0,
            x_star=x,
            f_star=fx,
            n_iters=n_iters,
            oracle_call_count=n_oracle,
            time_taken=t,
            history=history,
        )

    def _show_run_result(
        self,
        x: Vector,
        fx: Scalar,
        x0: Vector,
        n_iters: int | str,
        n_oracle: int | str,
        title: TextType | None = None,
    ) -> None:
        """Helper to format and display the values"""
        table = Table(title=title, title_justify="left", show_header=False)
        table.add_column(style="bold", justify="right")
        table.add_column()
        table.add_row("x0", format_value(x0, sep=", "))
        table.add_row("Iterations", str(n_iters))
        table.add_row("Oracle calls", str(n_oracle))
        table.add_section()
        show_solution(x, fx, table=table)


class ZeroOrderOptimiser[O: ZeroOrderOracle, T: ZeroOrderStepInfo[Any]](
    IterativeOptimiser[O, T], ABC
):
    @abstractmethod
    def step(self, info: T) -> T:
        raise NotImplementedError


class FirstOrderOptimiser[O: FirstOrderOracle, T: FirstOrderStepInfo[Any]](
    ZeroOrderOptimiser[O, T], ABC
):
    @abstractmethod
    def step(self, info: T) -> T:
        raise NotImplementedError


class LineSearchOptimiser[O: Oracle, T: LineSearchStepInfo[Any]](
    IterativeOptimiser[O, T]
):
    """
    A base template class for line search-based iterative optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `alpha_k` is the step length along the descent direction `p_k`.
    """

    def reset(self) -> Self:
        self.step_lengths: list[Scalar] = []
        self.step_directions: list[Vector] = []
        return super().reset()

    def direction(self, info: T) -> Vector:
        """
        Returns the descent direction `p_k` to move towards from `x_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific direction strategy.
        """
        raise NotImplementedError

    def step_length(self, info: T) -> Scalar:
        """
        Returns step length `alpha_k` to take along the descent direction `p_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific step length strategy.
        """
        raise NotImplementedError

    def step(self, info: T) -> T:
        p_k = self.direction(info)
        logger.debug(
            f"Step direction [bold magenta]\U0001d429{format_subscript(info.k)}[/] = {format_value(p_k, sep=', ')}"
        )
        info.direction = p_k
        self.step_directions.append(p_k)

        alpha_k = self.step_length(info)
        logger.debug(
            f"Step length [bold magenta]\U0001d770{format_subscript(info.k)}[/] = {alpha_k}"
        )
        info.alpha = alpha_k
        self.step_lengths.append(alpha_k)

        logger.debug(
            "Updating point: [bold yellow]"
            f"\U0001d431{format_subscript(info.k + 1)} \u2190 "
            f"\U0001d431{format_subscript(info.k)} \u002b "
            f"\U0001d770{format_subscript(info.k)} \u00d7 "
            f"\U0001d429{format_subscript(info.k)}[/]"
        )
        info_next = self.StepInfoClass(
            k=info.k + 1,
            x=Vector(info.x + alpha_k * p_k),
            oracle=info.oracle,
        )
        logger.debug(
            f"=> New point [bold magenta]\U0001d431{format_subscript(info_next.k)}[/] = {format_value(info_next.x, sep=', ')}"
        )
        return info_next

    @property
    def stopping(self) -> list[StoppingCriterionType[T]]:
        class StepLengthCriterion(StoppingCriterion[T]):
            def check(self, info: T) -> bool:
                tol: Scalar = 1e-16
                if info.alpha is None:
                    return False
                return abs(info.alpha) < tol

        return super().stopping + [StepLengthCriterion()]

    def plot_step_lengths(self) -> None:
        """Plot step lengths vs iterations for the best run."""
        plt.plot(self.step_lengths, marker="o", label=self.name)


class ZeroOrderLineSearchOptimiser[
    O: ZeroOrderOracle,
    T: ZeroOrderLineSearchStepInfo[Any],  # [FIXME] Type variance could remove Any
](LineSearchOptimiser[O, T], ZeroOrderOptimiser[O, T]):
    def _phi(
        self,
        info: T,
        alpha: Scalar,
        x: Optional[Vector] = None,
        direction: Optional[Vector] = None,
    ) -> Scalar:
        """`phi(alpha) = f(x + alpha * d)`"""
        x = info.ensure(x if x is not None else info.x)
        direction = info.ensure(direction if direction is not None else info.direction)
        return info.eval(Vector(x + alpha * direction))


class FirstOrderLineSearchOptimiser[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](ZeroOrderLineSearchOptimiser[O, T], FirstOrderOptimiser[O, T]):
    def _derphi(
        self,
        info: T,
        alpha: Scalar,
        x: Optional[Vector] = None,
        direction: Optional[Vector] = None,
    ) -> Scalar:
        """`phi'(alpha) = f'(x + alpha * d)^T d`"""
        x = info.ensure(x if x is not None else info.x)
        direction = info.ensure(direction if direction is not None else info.direction)
        return Scalar(info.grad(Vector(x + alpha * direction)).T @ direction)


class SteepestDescentDirectionMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T]):
    """
    A mixin class that provides the steepest descent direction strategy,
    i.e., the Cauchy direction, which is the negative gradient direction.

    `p_k = -f'(x_k)`
    """

    def direction(self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]) -> Vector:
        grad = info.dfx
        return -grad


class ExactLineSearchMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T]):
    """
    A mixin class that provides the exact line search step length strategy for convex quadratic functions.

    `alpha_k = - (f'(x_k)^T p_k) / (p_k^T Q p_k)`\\
    where `Q` is the symmetric positive definite Hessian matrix of the convex quadratic function.
    """

    def step_length(
        self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ) -> Scalar:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        d = info.ensure(info.direction)
        grad = info.dfx
        Q = info.oracle._oracle_f.Q

        numer = Scalar(grad.T @ d)
        denom = Scalar(d.T @ Q @ d)
        alpha = -numer / denom

        if alpha < 0:
            alpha = -alpha
            self.step_directions[-1] = -d

        return alpha


class NewtonDirectionMixin[
    O: SecondOrderOracle,
    T: SecondOrderLineSearchStepInfo[SecondOrderOracle],
](LineSearchOptimiser[O, T]):
    """
    A mixin class that provides the Newton direction strategy.

    `p_k = - (f''(x_k))^{-1} f'(x_k)`
    """

    def direction(self, info: T) -> Vector:
        grad = info.dfx
        hess = info.d2fx

        p_k = Vector(np.linalg.solve(hess, -grad))
        return p_k


class UnitStepLengthMixin[O: Oracle, T: LineSearchStepInfo[Any]](
    LineSearchOptimiser[O, T]
):
    """
    A mixin class that provides a unit step length strategy.

    `alpha_k = 1`
    """

    def step_length(self, info: T) -> Scalar:
        return 1.0


class QuasiNewtonOptimiser[
    O: FirstOrderOracle,
    T: QuasiNewtonStepInfo[FirstOrderOracle],
](UnitStepLengthMixin[O, T]):
    """
    A base template class for Quasi-Newton optimisation algorithms.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k = -H_k f'(x_k)` and `H_k` is the approximate inverse Hessian matrix.

    `s_{k+1} = x_{k+1} - x_k`\\
    `y_{k+1} = f'(x_{k+1}) - f'(x_k)`

    `s0` and `y0` are not defined, so they are left as `None` for iteration `k=0`.
    """

    @abstractmethod
    def hess_inv(self, info: T) -> Matrix:
        """
        Updates and returns the approximate inverse Hessian matrix `H_k`.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            info: An instance of `QuasiNewtonStepInfo` containing the current state of the algorithm.
        Returns:
            The updated approximate inverse Hessian matrix `H_k`.
        """
        raise NotImplementedError

    def direction(self, info: T) -> Vector:
        grad = info.dfx
        H = self.hess_inv(info)
        info.H = H
        pk = Vector(-H @ grad)
        return pk

    def step(self, info: T) -> T:
        info_next = super().step(info)
        info_next.H = info.H  # [FIXME]
        info_next.s = Vector(info_next.x - info.x)
        info_next.y = Vector(info_next.dfx - info.dfx)
        return info_next
