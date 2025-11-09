"""
Helpers
=======
src/cmo/helpers.py
"""

from typing import Optional

from .base import IterativeOptimiser
from .constraint import AbstractConstraint
from .functions import Function
from .oracle import AbstractOracle
from .problems import UnconstrainedProblem
from .stopping import StoppingCriterionType
from .types import Vector


def optimise(
    objective: Function,
    oracle: type[AbstractOracle],
    method: IterativeOptimiser,
    x0: Vector,
    constaint: Optional[AbstractConstraint] = None,
    criteria: Optional[StoppingCriterionType] = None,
    show_params: bool = True,
):
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
