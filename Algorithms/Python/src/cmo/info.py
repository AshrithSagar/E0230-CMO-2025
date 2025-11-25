"""
Info structures
=======
src/cmo/info.py
"""

import inspect
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .logging import logger
from .oracle import FirstOrderOracle, Oracle, SecondOrderOracle, ZeroOrderOracle
from .types import Matrix, Scalar, Vector
from .utils import format_subscript, format_time, format_value


@dataclass
class StepInfo[T: Oracle]:
    """A dataclass for the state of the algorithm at iteration `k`."""

    k: int
    """The current iteration number `k`."""

    x: Vector
    """The current point `x_k` in the input space."""

    oracle: T
    """The oracle function used to evaluate `f`."""

    def to_str(self, spacing: str | int = 2, prefix: str | int = 0) -> str:
        """Formatted string representation of the StepInfo object."""
        name: str = self.__class__.__name__
        s: str = " " * spacing if isinstance(spacing, int) else spacing
        p: str = " " * prefix if isinstance(prefix, int) else prefix
        repr: list[str] = []
        for f in fields(self):
            if f.name == "oracle":
                continue
            value = getattr(self, f.name)
            pfx = p + s + " " * 6 if f.name == "_d2fx" else ""
            repr.append(f"{f.name}={format_value(value, sep=', ', pfx=pfx)}")
        return f"{p}{name}(\n{p}{s}" + f",\n{p}{s}".join(repr) + f"\n{p})"

    def __str__(self) -> str:
        return self.to_str()

    def __format__(self, spec: str) -> str:
        if spec.isdigit():
            return self.to_str(spacing=int(spec))
        return self.to_str()

    def ensure[A](
        self,
        attr: Optional[A],
        fallback: Optional[A] = None,
        message: Optional[str] = None,
    ) -> A:
        """
        Ensure that the attribute of type `A` is not `None`.

        Parameters
        ----------
        attr : Optional[A]
            The attribute to check.
        fallback : Optional[A], optional
            The fallback value to return if `attr` is `None`.
            Defaults to `None`, in which case an error is raised.
        message : Optional[str], optional
            The message to raise in the ValueError if `attr` is `None` and no fallback is provided.
            Defaults to `None`, in which case a generic message is used.
        """
        if attr is None:
            if fallback is not None:
                return fallback
            if message is not None:
                raise ValueError(message)
            expr = "value"
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                try:
                    code_ctx = inspect.getframeinfo(caller).code_context
                    if code_ctx:
                        code = code_ctx[0]
                        expr = code.split("ensure(")[1].split(")")[0].strip()
                except Exception:
                    expr = "value"
            raise ValueError(f"{expr} is None.")
        return attr


@dataclass
class ZeroOrderStepInfo[T: ZeroOrderOracle](StepInfo[T]):
    _fx: Optional[Scalar] = field(init=False, default=None)
    """Internal function value at `x_k`."""

    @property
    def fx(self) -> Scalar:
        """The function value `f(x_k)` at the current point."""
        if self._fx is None:
            self._fx = self.eval(self.x)
            logger.debug(
                f"=> Function value [bold magenta]\U0001d453(\U0001d431{format_subscript(self.k)})[/] = {format_value(self._fx)}"
            )
        return self._fx

    def eval(self, x: Vector) -> Scalar:
        logger.debug(
            f"Computing function value at \U0001d431 = {format_value(x, sep=', ')}"
        )
        return self.oracle.eval(x)


@dataclass
class FirstOrderStepInfo[T: FirstOrderOracle](ZeroOrderStepInfo[T]):
    _dfx: Optional[Vector] = field(init=False, default=None)
    """Internal gradient at `x_k`."""

    @property
    def dfx(self) -> Vector:
        """The gradient `f'(x_k)` at the current point."""
        if self._dfx is None:
            self._dfx = self.grad(self.x)
            logger.debug(
                f"=> Gradient [bold magenta]\U0001d6c1\U0001d453(\U0001d431{format_subscript(self.k)})[/] = {format_value(self._dfx, sep=', ')}"
            )
        return self._dfx

    def grad(self, x: Vector) -> Vector:
        logger.debug(f"Computing gradient at \U0001d431 = {format_value(x, sep=', ')}")
        return self.oracle.grad(x)


@dataclass
class SecondOrderStepInfo[T: SecondOrderOracle](FirstOrderStepInfo[T]):
    _d2fx: Optional[Matrix] = field(init=False, default=None)
    """Internal Hessian at `x_k`."""

    @property
    def d2fx(self) -> Matrix:
        """The Hessian `f''(x_k)` at the current point."""
        if self._d2fx is None:
            self._d2fx = self.hess(self.x)
            logger.debug(
                f"=> Hessian [bold magenta]\U0001d6c1\u00b2\U0001d453(\U0001d431{format_subscript(self.k)})[/] = {format_value(self._d2fx, sep=', ')}"
            )
        return self._d2fx

    def hess(self, x: Vector) -> Matrix:
        logger.debug(f"Computing Hessian at \U0001d431 = {format_value(x, sep=', ')}")
        return self.oracle.hess(x)


@dataclass
class LineSearchStepInfo[T: Oracle](StepInfo[T]):
    direction: Optional[Vector] = None
    alpha: Optional[Scalar] = None


@dataclass
class ZeroOrderLineSearchStepInfo[T: ZeroOrderOracle](
    ZeroOrderStepInfo[T], LineSearchStepInfo[T]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class FirstOrderLineSearchStepInfo[T: FirstOrderOracle](
    FirstOrderStepInfo[T], ZeroOrderLineSearchStepInfo[T]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class SecondOrderLineSearchStepInfo[T: SecondOrderOracle](
    SecondOrderStepInfo[T], FirstOrderLineSearchStepInfo[T]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class QuasiNewtonStepInfo[T: FirstOrderOracle](FirstOrderLineSearchStepInfo[T]):
    H: Optional[Matrix] = None
    """Approximate inverse Hessian matrix at iteration `k`."""

    s: Optional[Vector] = None
    """
    Step taken after the previous iteration, i.e., iteration `k-1`.

    `s_k = x_k - x_{k-1}`
    """

    y: Optional[Vector] = None
    """
    Gradient difference after the previous iteration, i.e., iteration `k-1`.

    `y_k = f'(x_k) - f'(x_{k-1})`
    """


@dataclass
class ActiveSetStepInfo[T: FirstOrderOracle](FirstOrderLineSearchStepInfo[T]):
    W: Optional[list[int]] = None
    """Indices of the active constraints at iteration `k`."""

    mu: Optional[Vector] = None
    """The Lagrange multipliers associated with the active constraints at iteration `k`."""

    blocking: Optional[int] = None
    """Index of the blocking constraint, if any."""

    relax: Optional[int] = None
    """Index of the constraint relaxed, if any."""


@dataclass(kw_only=True)
class RunInfo[O: Oracle, T: StepInfo[Any]]:
    """
    A dataclass that holds information about the optimisation run.
    - 'x0': Initial point.
    - 'x_star': Estimated minimum point.
    - 'f_star': Function value at the estimated minimum point.
    - 'n_iters': Number of iterations performed.
    - 'oracle_call_count': Total number of oracle calls made.
    - 'time_taken': Total time taken for the optimisation run.
    - 'history': List of `x` values at each iteration.
    """

    x0: Vector
    x_star: Vector
    f_star: Scalar
    n_iters: int
    oracle_call_count: int
    time_taken: Scalar
    history: list[T]

    def to_str(self, spacing: str | int = 2) -> str:
        name: str = self.__class__.__name__
        s: str = " " * spacing if isinstance(spacing, int) else spacing
        repr: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "history":
                continue
            if f.name == "time_taken":
                repr.append(f"{f.name}={format_time(value)}")
            else:
                repr.append(f"{f.name}={format_value(value, sep=', ')}")
        return (
            f"{name}(\n{s}"
            + f",\n{s}".join(repr)
            + f",\n{s}history=[\n"
            + f"{',\n'.join(step.to_str(prefix=spacing * 2, spacing=spacing) for step in self.history)}\n{s}]\n)"
        )

    def __str__(self) -> str:
        return self.to_str()

    def __format__(self, spec: str) -> str:
        if spec.isdigit():
            return self.to_str(spacing=int(spec))
        return self.to_str()
