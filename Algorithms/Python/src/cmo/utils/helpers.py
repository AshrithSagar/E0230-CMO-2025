"""
Helpers
=======
src/cmo/utils/helpers.py
"""

from typing import Any, Optional, Type

from cmo.constrained.constraint import AbstractConstraint
from cmo.core.base import IterativeOptimiser
from cmo.core.info import RunInfo, StepInfo
from cmo.core.oracle import Oracle
from cmo.core.stopping import StoppingCriterionType
from cmo.functions.protocol import ZeroOrderFunctionProto
from cmo.problems import UnconstrainedProblem
from cmo.types import Vector


def optimise[F: ZeroOrderFunctionProto](
    objective: F,
    oracle: Type[Oracle[F]],
    method: IterativeOptimiser[StepInfo[Oracle[F]]],
    x0: Vector,
    constaint: Optional[AbstractConstraint[Any]] = None,
    criteria: Optional[StoppingCriterionType[StepInfo[Oracle[F]]]] = None,
    show_params: bool = True,
) -> Optional[RunInfo[StepInfo[Oracle[F]]]]:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
