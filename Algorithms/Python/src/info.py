"""
src/info.py
=======
Info structures
"""

from dataclasses import dataclass, field
from typing import Generic, Optional, TypedDict, TypeVar

from .oracle import AbstractOracle, FirstOrderOracle, SecondOrderOracle, ZeroOrderOracle
from .types import floatMat, floatVec

TOracle = TypeVar("TOracle", bound=AbstractOracle)
"""Generic type variable for Oracle subclasses."""

TZeroOrderOracle = TypeVar("TZeroOrderOracle", bound=ZeroOrderOracle)
"""Generic type variable for ZeroOrderOracle subclasses."""

TFirstOrderOracle = TypeVar("TFirstOrderOracle", bound=FirstOrderOracle)
"""Generic type variable for FirstOrderOracle subclasses."""

TSecondOrderOracle = TypeVar("TSecondOrderOracle", bound=SecondOrderOracle)
"""Generic type variable for SecondOrderOracle subclasses."""


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
class LineSearchStepInfo(StepInfo):
    direction: Optional[floatVec] = None
    alpha: Optional[float] = None


@dataclass
class FirstOrderLineSearchStepInfo(FirstOrderStepInfo, LineSearchStepInfo):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class SecondOrderLineSearchStepInfo(SecondOrderStepInfo, LineSearchStepInfo):
    pass  # All attributes and methods are provided by the parent classes


@dataclass
class QuasiNewtonStepInfo(FirstOrderLineSearchStepInfo):
    H: Optional[floatMat] = None
    """Approximate inverse Hessian matrix at iteration `k`."""


class RunInfo(TypedDict):
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
    history: list[floatVec]
    oracle_call_count: int
    time_taken: float
