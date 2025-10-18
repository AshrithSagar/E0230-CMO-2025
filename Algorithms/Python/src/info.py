"""
src/info.py
=======
Info structures
"""

from dataclasses import dataclass, field
from typing import Optional, TypedDict, TypeVar

from .oracle import AbstractOracle, FirstOrderOracle, SecondOrderOracle, ZeroOrderOracle
from .types import floatMat, floatVec

TStepInfo = TypeVar("TStepInfo", bound="StepInfo")
"""Generic type variable for StepInfo subclasses."""


@dataclass
class StepInfo:
    """A dataclass for the state of the algorithm at iteration `k`."""

    x: floatVec
    """The current point `x_k` in the input space."""

    k: int
    """The current iteration number `k`."""

    oracle: AbstractOracle
    """The oracle function used to evaluate `f`."""

    # Internal values
    _fx: Optional[float] = field(init=False, default=None)
    _dfx: Optional[floatVec] = field(init=False, default=None)
    _d2fx: Optional[floatMat] = field(init=False, default=None)

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


@dataclass
class LineSearchStepInfo(StepInfo):
    direction: Optional[floatVec] = None
    alpha: Optional[float] = None


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
