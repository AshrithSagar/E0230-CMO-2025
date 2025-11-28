"""
Helpers
=======
src/cmo/utils/helpers.py
"""

from typing import Any, Optional

from cmo.constrained.constraint import AbstractConstraint
from cmo.core.base import IterativeOptimiser
from cmo.core.info import RunInfo, StepInfo
from cmo.core.oracle import Oracle
from cmo.core.stopping import StoppingCriterionType
from cmo.functions import Function
from cmo.problems import UnconstrainedProblem
from cmo.types import Vector


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
