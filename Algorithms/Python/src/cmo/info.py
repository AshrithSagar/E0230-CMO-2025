"""
Info structures
=======
src/cmo/info.py
"""

from dataclasses import dataclass, field, fields
from typing import Optional

from .oracle import AbstractOracle, FirstOrderOracle, SecondOrderOracle, ZeroOrderOracle
from .types import floatMat, floatVec
from .utils import format_float, format_time


@dataclass
class StepInfo[T: AbstractOracle]:
    """A dataclass for the state of the algorithm at iteration `k`."""

    k: int
    """The current iteration number `k`."""

    x: floatVec
    """The current point `x_k` in the input space."""

    oracle: T
    """The oracle function used to evaluate `f`."""

    def __str__(self, spacing: str | int = 2, prefix: str | int = 0) -> str:
        """String representation of the StepInfo object."""
        name: str = self.__class__.__name__
        s: str = " " * spacing if isinstance(spacing, int) else spacing
        p: str = " " * prefix if isinstance(prefix, int) else prefix
        repr: list[str] = []
        for f in fields(self):
            if f.name == "oracle":
                continue
            value = getattr(self, f.name)
            pfx = p + s + " " * 6 if f.name == "_d2fx" else ""
            repr.append(f"{f.name}={format_float(value, sep=', ', pfx=pfx)}")
        return f"{p}{name}(\n{p}{s}" + f",\n{p}{s}".join(repr) + f"\n{p})"


@dataclass
class ZeroOrderStepInfo[T: ZeroOrderOracle](StepInfo[T]):
    _fx: Optional[float] = field(init=False, default=None)
    """Internal function value at `x_k`."""

    @property
    def fx(self) -> float:
        """The function value `f(x_k)` at the current point."""
        if self._fx is None:
            self._fx = self.eval(self.x)
        return self._fx

    def eval(self, x: floatVec) -> float:
        return self.oracle.eval(x)


@dataclass
class FirstOrderStepInfo[T: FirstOrderOracle](ZeroOrderStepInfo[T]):
    _dfx: Optional[floatVec] = field(init=False, default=None)
    """Internal gradient at `x_k`."""

    @property
    def dfx(self) -> floatVec:
        """The gradient `f'(x_k)` at the current point."""
        if self._dfx is None:
            self._dfx = self.grad(self.x)
        return self._dfx

    def grad(self, x: floatVec) -> floatVec:
        return self.oracle.grad(x)


@dataclass
class SecondOrderStepInfo[T: SecondOrderOracle](FirstOrderStepInfo[T]):
    _d2fx: Optional[floatMat] = field(init=False, default=None)
    """Internal Hessian at `x_k`."""

    @property
    def d2fx(self) -> floatMat:
        """The Hessian `f''(x_k)` at the current point."""
        if self._d2fx is None:
            self._d2fx = self.hess(self.x)
        return self._d2fx

    def hess(self, x: floatVec) -> floatMat:
        return self.oracle.hess(x)


@dataclass
class LineSearchStepInfo[T: AbstractOracle](StepInfo[T]):
    direction: Optional[floatVec] = None
    alpha: Optional[float] = None


@dataclass
class FirstOrderLineSearchStepInfo[T: FirstOrderOracle](
    FirstOrderStepInfo[T], LineSearchStepInfo[T]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class SecondOrderLineSearchStepInfo[T: SecondOrderOracle](
    SecondOrderStepInfo[T], LineSearchStepInfo[T]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class QuasiNewtonStepInfo[T: FirstOrderOracle](FirstOrderLineSearchStepInfo[T]):
    H: Optional[floatMat] = None
    """Approximate inverse Hessian matrix at iteration `k`."""

    s: Optional[floatVec] = None
    """
    Step taken after the previous iteration, i.e., iteration `k-1`.

    `s_k = x_k - x_{k-1}`
    """

    y: Optional[floatVec] = None
    """
    Gradient difference after the previous iteration, i.e., iteration `k-1`.

    `y_k = f'(x_k) - f'(x_{k-1})`
    """


@dataclass
class ActiveSetStepInfo[T: FirstOrderOracle](FirstOrderLineSearchStepInfo[T]):
    W: Optional[list[int]] = None
    """Indices of the active constraints at iteration `k`."""

    mu: Optional[floatVec] = None
    """The Lagrange multipliers associated with the active constraints at iteration `k`."""

    blocking: Optional[int] = None
    """Index of the blocking constraint, if any."""

    relax: Optional[int] = None
    """Index of the constraint relaxed, if any."""


@dataclass(kw_only=True)
class RunInfo[T: StepInfo]:
    """
    A dataclass that holds information about the optimisation run.
    - 'x0': Initial point.
    - 'x_star': Estimated minimum point.
    - 'f_star': Function value at the estimated minimum point.
    - 'n_iters': Number of iterations performed.
    - 'history': List of `x` values at each iteration.
    - 'oracle_call_count': Total number of oracle calls made.
    - 'time_taken': Total time taken for the optimisation run.
    """

    x0: floatVec
    x_star: floatVec
    f_star: float
    n_iters: int
    history: list[T]
    oracle_call_count: int
    time_taken: float

    def __str__(self, spacing: str | int = 2) -> str:
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
                repr.append(f"{f.name}={format_float(value, sep=', ')}")
        return (
            f"{name}(\n{s}"
            + f",\n{s}".join(repr)
            + f",\n{s}history=[\n"
            + f"{',\n'.join(step.__str__(prefix=spacing * 2, spacing=spacing) for step in self.history)}\n{s}]\n)"
        )
