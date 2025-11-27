"""
Helpers
=======
src/cmo/helpers.py
"""

from typing import Any, Optional

from ..constrained.constraint import AbstractConstraint
from ..core.base import IterativeOptimiser
from ..core.info import RunInfo, StepInfo
from ..core.oracle import Oracle
from ..core.stopping import StoppingCriterionType
from ..functions import Function
from ..problems import UnconstrainedProblem
from ..types import Vector


def optimise[O: Oracle, T: StepInfo[Any]](
    objective: Function,
    oracle: type[O],
    method: IterativeOptimiser[O, T],
    x0: Vector,
    constaint: Optional[AbstractConstraint[Any]] = None,
    criteria: Optional[StoppingCriterionType[T]] = None,
    show_params: bool = True,
) -> Optional[RunInfo[O, T]]:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
