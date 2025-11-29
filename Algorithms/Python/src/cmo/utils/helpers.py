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
from cmo.functions.protocol import FunctionProto
from cmo.problems import UnconstrainedProblem
from cmo.types import Vector


def optimise[F: FunctionProto, O: Oracle[Any], S: StepInfo[Any, Any]](
    objective: F,
    oracle: type[O],
    method: IterativeOptimiser[F, O, S],
    x0: Vector,
    constaint: Optional[AbstractConstraint[Any]] = None,
    criteria: Optional[StoppingCriterionType[F, O, S]] = None,
    show_params: bool = True,
) -> Optional[RunInfo[F, O, S]]:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
