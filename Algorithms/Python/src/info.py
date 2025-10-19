"""
src/info.py
=======
Info structures
"""

from dataclasses import dataclass, field
from typing import Generic, Optional, TypedDict, TypeVar

from .oracle import TFirstOrderOracle, TOracle, TSecondOrderOracle, TZeroOrderOracle
from .types import floatMat, floatVec

TStepInfo = TypeVar("TStepInfo", bound="StepInfo")
"""Generic type variable for StepInfo subclasses."""

TLineSearchStepInfo = TypeVar("TLineSearchStepInfo", bound="LineSearchStepInfo")
"""Generic type variable for LineSearchStepInfo subclasses."""


@dataclass
class StepInfo(Generic[TOracle]):
    """A dataclass for the state of the algorithm at iteration `k`."""

    x: floatVec
    """The current point `x_k` in the input space."""

    k: int
    """The current iteration number `k`."""

    oracle: TOracle
    """The oracle function used to evaluate `f`."""


@dataclass
class ZeroOrderStepInfo(StepInfo[TZeroOrderOracle]):
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
class FirstOrderStepInfo(ZeroOrderStepInfo[TFirstOrderOracle]):
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
class SecondOrderStepInfo(FirstOrderStepInfo[TSecondOrderOracle]):
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
class LineSearchStepInfo(StepInfo[TOracle]):
    direction: Optional[floatVec] = None
    alpha: Optional[float] = None


@dataclass
class FirstOrderLineSearchStepInfo(
    FirstOrderStepInfo[TFirstOrderOracle], LineSearchStepInfo[TFirstOrderOracle]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class SecondOrderLineSearchStepInfo(
    SecondOrderStepInfo[TSecondOrderOracle], LineSearchStepInfo[TSecondOrderOracle]
):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class QuasiNewtonStepInfo(FirstOrderLineSearchStepInfo[TFirstOrderOracle]):
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


class RunInfo(TypedDict, Generic[TStepInfo]):
    """
    Dictionary type for storing run information of an optimiser.
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
    history: list[TStepInfo]
    oracle_call_count: int
    time_taken: float
