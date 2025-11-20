"""
Helpers
=======
src/cmo/helpers.py
"""

from typing import Optional

from .base import IterativeOptimiser
from .constraint import AbstractConstraint
from .functions import Function
from .info import RunInfo, StepInfo
from .oracle import AbstractOracle
from .problems import UnconstrainedProblem
from .stopping import StoppingCriterionType
from .types import Vector


def optimise[T: StepInfo](
    objective: Function,
    oracle: type[AbstractOracle],
    method: IterativeOptimiser[T],
    x0: Vector,
    constaint: Optional[AbstractConstraint] = None,
    criteria: Optional[StoppingCriterionType] = None,
    show_params: bool = True,
) -> Optional[RunInfo[T]]:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
