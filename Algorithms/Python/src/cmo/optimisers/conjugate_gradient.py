"""
Conjugate Gradient methods
=======
src/cmo/optimisers/conjugate_gradient.py
"""

from typing import Any, Self

from ..base import ExactLineSearchMixin, FirstOrderLineSearchOptimiser
from ..functions import ConvexQuadratic
from ..info import FirstOrderLineSearchStepInfo
from ..oracle import FirstOrderOracle
from ..types import Scalar, Vector


class ConjugateDirectionMethod(
    FirstOrderLineSearchOptimiser[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ]
):
    """
    Linear conjugate direction method for convex quadratic functions.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    def __init__(self, directions: list[Vector], **kwargs: Any) -> None:
        super().__init__(directions=directions, **kwargs)
        self.directions: list[Vector] = directions
        self.line_search = ExactLineSearchMixin[
            FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
        ]()

    def reset(self) -> Self:
        self.line_search.reset()
        return super().reset()

    def direction(self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]) -> Vector:
        k = info.k
        if k < len(self.directions):
            direction = self.directions[k]
            self.line_search.step_directions.append(direction)
            return direction
        else:
            raise IndexError(f"No more directions available for iteration {k}.")

    def step_length(
        self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ) -> Scalar:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        alpha = self.line_search.step_length(info)
        self.step_directions[-1] = self.line_search.step_directions[-1]
        return alpha


class ConjugateGradientMethod(
    FirstOrderLineSearchOptimiser[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ]
):
    """
    Linear conjugate gradient method for convex quadratic functions.

    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.line_search = ExactLineSearchMixin[
            FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
        ]()

    def reset(self) -> Self:
        self.line_search.reset()
        return super().reset()

    def direction(self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]) -> Vector:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        k = info.k
        grad = info.dfx

        if k == 0:
            self.rTr_prev = Scalar(grad.T @ grad)
            direction = -grad
        else:
            rTr = Scalar(grad.T @ grad)
            beta = rTr / self.rTr_prev
            direction = Vector(grad + beta * self.step_directions[-1])
            self.rTr_prev = rTr

        self.line_search.step_directions.append(direction)
        return direction

    def step_length(
        self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ) -> Scalar:
        alpha = self.line_search.step_length(info)
        self.step_directions[-1] = self.line_search.step_directions[-1]
        return alpha
