"""
Optimisation algorithms.
=======
src/cmo/optimisers.py

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

from abc import ABC
from typing import Any, Self

import numpy as np

from .base import (
    ExactLineSearchMixin,
    FirstOrderLineSearchOptimiser,
    NewtonDirectionMixin,
    QuasiNewtonOptimiser,
    SteepestDescentDirectionMixin,
    UnitStepLengthMixin,
)
from .functions import ConvexQuadratic
from .info import (
    FirstOrderLineSearchStepInfo,
    QuasiNewtonStepInfo,
    SecondOrderLineSearchStepInfo,
)
from .logging import logger
from .oracle import FirstOrderOracle, SecondOrderOracle
from .types import Matrix, Scalar, Vector, dtype


class ArmijoMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T], ABC):
    """
    A mixin class that provides the forward-expansion Armijo line search step length strategy.\\
    Increase alpha until Armijo condition holds (or until safe cap).

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k`
    """

    def __init__(
        self,
        c: Scalar = 1e-4,
        alpha_min: Scalar = 1e-14,
        alpha_start: Scalar = 0.0,
        alpha_step: Scalar = 1e-1,
        alpha_stop: Scalar = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            c=c,
            alpha_min=alpha_min,
            alpha_start=alpha_start,
            alpha_step=alpha_step,
            alpha_stop=alpha_stop,
            **kwargs,
        )

        self.c = Scalar(c)
        """Armijo parameter"""
        self.alpha_min = Scalar(alpha_min)
        self.alpha_start = Scalar(alpha_start)
        self.alpha_step = Scalar(alpha_step)
        self.alpha_stop = Scalar(alpha_stop)

    def reset(self) -> Self:
        assert 0 < self.c < 1, "c must be in (0, 1)"
        return super().reset()

    def step_length(self, info: T) -> Scalar:
        d = info.ensure(info.direction)
        f = info.fx
        grad = info.dfx
        derphi0 = Scalar(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # Forward expansion
        alpha_prev: Scalar | None = None
        for _alpha in np.arange(
            self.alpha_start,
            self.alpha_stop + self.alpha_step,
            self.alpha_step,
            dtype=dtype,
        ):
            alpha = Scalar(_alpha)
            f_new = self._phi(info, alpha)
            if f_new <= f + self.c * alpha * derphi0:
                alpha_prev = alpha
            else:
                break
        if alpha_prev is not None:
            return alpha_prev
        else:
            return self.alpha_min


class BacktrackingMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T], ABC):
    """
    A mixin class for the standard backtracking Armijo (decreasing alpha).
    """

    def __init__(
        self,
        c: Scalar = 1e-4,
        beta: Scalar = 0.5,
        alpha_init: Scalar = 1.0,
        alpha_min: Scalar = 1e-14,
        alpha_max: Scalar = 1e6,
        maxiter: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            c=c,
            beta=beta,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            maxiter=maxiter,
            **kwargs,
        )

        self.c = Scalar(c)
        """Armijo parameter"""
        self.beta = Scalar(beta)
        """Step length reduction factor"""
        self.alpha_init = Scalar(alpha_init)
        self.alpha_min = Scalar(alpha_min)
        self.alpha_max = Scalar(alpha_max)
        self.maxiter = int(maxiter)

    def reset(self) -> Self:
        assert 0 < self.c < 1, "c must be in (0, 1)"
        return super().reset()

    def step_length(self, info: T) -> Scalar:
        d = info.ensure(info.direction)
        f = info.fx
        grad = info.dfx
        derphi0 = Scalar(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        alpha = self.alpha_init
        for _ in range(self.maxiter):
            f_new = self._phi(info, alpha)
            if f_new <= f + self.c * alpha * derphi0:
                return alpha
            alpha *= self.beta
            if alpha < self.alpha_min:
                return self.alpha_min
        return alpha


class ArmijoGoldsteinMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T], ABC):
    """
    A mixin class for Armijo-Goldstein via expansion to bracket and then bisection.

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k` (Armijo)\\
    `f(x_k + alpha_k * p_k) >= f(x_k) + (1 - c) * alpha_k * f'(x_k)^T p_k` (Goldstein)
    """

    def __init__(
        self,
        c: Scalar = 1e-4,
        beta: Scalar = 0.5,
        alpha_init: Scalar = 1.0,
        alpha_min: Scalar = 1e-14,
        alpha_max: Scalar = 1e6,
        maxiter: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            c=c,
            beta=beta,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            maxiter=maxiter,
            **kwargs,
        )

        self.c = Scalar(c)
        """Armijo-Goldstein parameter"""
        self.beta = Scalar(beta)
        """Step length reduction factor"""
        self.alpha_init = Scalar(alpha_init)
        self.alpha_min = Scalar(alpha_min)
        self.alpha_max = Scalar(alpha_max)
        self.maxiter = int(maxiter)

    def reset(self) -> Self:
        assert 0 < self.c < 0.5, "c must be in (0, 0.5)"
        return super().reset()

    def step_length(self, info: T) -> Scalar:
        d = info.ensure(info.direction)
        f = info.fx
        grad = info.dfx
        derphi0 = Scalar(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # If initial alpha already satisfies both, return it
        alpha = self.alpha_init
        f_new = self._phi(info, alpha)
        if (f_new <= f + self.c * alpha * derphi0) and (
            f_new >= f + (1 - self.c) * alpha * derphi0
        ):
            return alpha

        # Expand to find an interval [alpha_lo, alpha_hi] where Armijo condition is satisfied at alpha_hi
        alpha_lo = 0.0
        alpha_hi = alpha
        for _ in range(self.maxiter):
            phi_hi = self._phi(info, alpha_hi)
            if phi_hi <= f + self.c * alpha_hi * derphi0:
                break
            alpha_hi *= self.beta
            if alpha_hi > self.alpha_max:
                break

        # Bisect between alpha_lo and alpha_hi until Goldstein condition hold
        for _ in range(self.maxiter):
            alpha_mid = 0.5 * (alpha_lo + alpha_hi)
            phi_mid = self._phi(info, alpha_mid)
            if (phi_mid <= f + self.c * alpha_mid * derphi0) and (
                phi_mid >= f + (1 - self.c) * alpha_mid * derphi0
            ):
                return alpha_mid
            # If phi_mid is too large -> need smaller step (move hi)
            if phi_mid > f + self.c * alpha_mid * derphi0:
                alpha_hi = alpha_mid
            else:
                # phi_mid < lower bound, it may violate upper bound -> move lo
                alpha_lo = alpha_mid
            if abs(alpha_hi - alpha_lo) < self.alpha_min:
                break
        return 0.5 * (alpha_lo + alpha_hi)


class StrongWolfeMixin[
    O: FirstOrderOracle,
    T: FirstOrderLineSearchStepInfo[FirstOrderOracle],
](FirstOrderLineSearchOptimiser[O, T], ABC):
    """
    A mixin class for the strong Wolfe line search using bracket + zoom (Nocedal & Wright).

    `phi(alpha_k) <= phi(0) + c1 * alpha_k * phi'(0)` (Armijo)\\
    `|phi'(alpha_k)| <= c2 * |phi'(0)|` (Strong curvature)\\
    where\\
    `phi(alpha_k) = f(x_k + alpha_k * p_k)`,\\
    `phi'(alpha_k) = f'(x_k + alpha_k * p_k)^T p_k`.
    """

    def __init__(
        self,
        c1: Scalar = 1e-4,
        c2: Scalar = 0.9,
        beta: Scalar = 0.5,
        alpha_init: Scalar = 1.0,
        alpha_min: Scalar = 1e-14,
        alpha_max: Scalar = 1e6,
        maxiter: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            c1=c1,
            c2=c2,
            beta=beta,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            maxiter=maxiter,
            **kwargs,
        )

        self.c1 = Scalar(c1)
        """Armijo parameter"""
        self.c2 = Scalar(c2)
        """Curvature parameter"""
        self.beta = Scalar(beta)
        """Step length reduction factor"""
        self.alpha_init = Scalar(alpha_init)
        self.alpha_min = Scalar(alpha_min)
        self.alpha_max = Scalar(alpha_max)
        self.maxiter = int(maxiter)

    def reset(self) -> Self:
        assert 0 < self.c1 < self.c2 < 1, "0 < c1 < c2 < 1 must be satisfied"
        return super().reset()

    def step_length(self, info: T) -> Scalar:
        d = info.ensure(info.direction)
        f = info.fx
        grad = info.dfx
        derphi0 = Scalar(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        phi0 = f
        alpha = self.alpha_init
        phi_prev = phi0
        alpha_prev = 0.0

        for i in range(self.maxiter):
            phi_a = self._phi(info, alpha)
            derphi_a = self._derphi(info, alpha)

            # Check Armijo
            if (phi_a > phi0 + self.c1 * alpha * derphi0) or (
                i > 0 and phi_a >= phi_prev
            ):
                # bracket found between alpha_prev and alpha
                return self._zoom(info, alpha_prev, alpha, phi0, derphi0)
            # Check strong Wolfe
            if abs(derphi_a) <= self.c2 * abs(derphi0):
                return alpha
            # If derivative is positive, bracket and zoom
            if derphi_a >= 0:
                return self._zoom(info, alpha, alpha_prev, phi0, derphi0)
            # Otherwise increase alpha (extrapolate)
            alpha_prev = alpha
            phi_prev = phi_a
            alpha = alpha * self.beta
        return alpha  # Fallback

    def _zoom(
        self,
        info: T,
        alpha_lo: Scalar,
        alpha_hi: Scalar,
        phi0: Scalar,
        derphi0: Scalar,
        maxiter: int = 50,
    ) -> Scalar:
        """
        Zoom procedure as in Nocedal & Wright (uses safe bisection interpolation).
        Returns an alpha that satisfies strong Wolfe (if found), otherwise the best found.
        """
        phi_lo = self._phi(info, alpha_lo)
        _derphi_lo = self._derphi(info, alpha_lo)
        for _ in range(maxiter):
            alpha_j = 0.5 * (alpha_lo + alpha_hi)  # safe midpoint
            phi_j = self._phi(info, alpha_j)
            derphi_j = self._derphi(info, alpha_j)

            # Armijo condition
            if (phi_j > phi0 + self.c1 * alpha_j * derphi0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j
            else:
                # Check strong Wolfe condition (absolute derivative)
                if abs(derphi_j) <= self.c2 * abs(derphi0):
                    return alpha_j
                # If derivative has same sign as _derphi_lo, shrink interval
                if derphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
                phi_lo = phi_j
                _derphi_lo = derphi_j
            if abs(alpha_hi - alpha_lo) < 1e-14:
                break
        # fallback: return midpoint
        return 0.5 * (alpha_lo + alpha_hi)


class GradientDescent(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ]
):
    """
    Standard gradient descent.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    def __init__(self, lr: Scalar = 1e-3, **kwargs: Any) -> None:
        super().__init__(lr=lr, **kwargs)
        self.lr = Scalar(lr)
        """Learning rate (step length)"""

    def step_length(
        self, info: FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ) -> Scalar:
        return self.lr


class GradientDescentExactLineSearch(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
    ExactLineSearchMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
):
    """
    Gradient descent with exact line search for convex quadratic functions.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`\\
    where `alpha_k = (f'(x_k)^T f'(x_k)) / (f'(x_k)^T Q f'(x_k))`
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    pass  # All methods are provided by the mixins


class GradientDescentArmijo(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
    ArmijoMixin[FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]],
):
    """
    Forward-expansion Armijo line search:\\
    Increase alpha until Armijo condition holds (or until safe cap).

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k`
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    pass  # All methods are provided by the mixins


class GradientDescentBacktracking(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
    BacktrackingMixin[FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]],
):
    """
    Standard backtracking Armijo (decreasing alpha).
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    pass  # All methods are provided by the mixins


class GradientDescentArmijoGoldstein(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
    ArmijoGoldsteinMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
):
    """
    Armijo-Goldstein via expansion to bracket and then bisection.

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k` (Armijo)\\
    `f(x_k + alpha_k * p_k) >= f(x_k) + (1 - c) * alpha_k * f'(x_k)^T p_k` (Goldstein)
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    pass  # All methods are provided by the mixins


class GradientDescentWolfe(
    SteepestDescentDirectionMixin[
        FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]
    ],
    StrongWolfeMixin[FirstOrderOracle, FirstOrderLineSearchStepInfo[FirstOrderOracle]],
):
    """
    Strong Wolfe line search using bracket + zoom (Nocedal & Wright).

    `phi(alpha_k) <= phi(0) + c1 * alpha_k * phi'(0)` (Armijo)\\
    `|phi'(alpha_k)| <= c2 * |phi'(0)|` (Strong curvature)\\
    where\\
    `phi(alpha_k) = f(x_k + alpha_k * p_k)`,\\
    `phi'(alpha_k) = f'(x_k + alpha_k * p_k)^T p_k`.
    """

    StepInfoClass = FirstOrderLineSearchStepInfo[FirstOrderOracle]

    pass  # All methods are provided by the mixins


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


class NewtonMethod(
    NewtonDirectionMixin[
        SecondOrderOracle, SecondOrderLineSearchStepInfo[SecondOrderOracle]
    ],
    UnitStepLengthMixin[
        SecondOrderOracle, SecondOrderLineSearchStepInfo[SecondOrderOracle]
    ],
):
    """
    Standard Newton's method.

    `x_{k+1} = x_k - [f''(x_k)]^{-1} f'(x_k)`
    """

    StepInfoClass = SecondOrderLineSearchStepInfo[SecondOrderOracle]

    pass  # All methods are provided by the mixins


class SR1Update(
    QuasiNewtonOptimiser[FirstOrderOracle, QuasiNewtonStepInfo[FirstOrderOracle]]
):
    """
    Symmetric Rank-One (SR1) update for Hessian inverse approximation.

    `H_{k+1} = H_k + ((s_k - H_k y_k)(s_k - H_k y_k)^T) / ((s_k - H_k y_k)^T y_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[FirstOrderOracle]

    def hess_inv(self, info: QuasiNewtonStepInfo[FirstOrderOracle]) -> Matrix:
        if info.k == 0:
            if info.H is None:
                logger.debug("Initialising Hessian inverse approximation to identity.")
                info.H = Matrix(np.eye(info.x.shape[0], dtype=dtype))
            return info.H
        else:
            logger.debug("Performing SR1 Hessian inverse update.")
            H, s, y = info.ensure(info.H), info.ensure(info.s), info.ensure(info.y)
            u = Vector(s - H @ y)
            return Matrix(H + np.outer(u, u) / Scalar(u.T @ y))


class DFPUpdate(
    QuasiNewtonOptimiser[FirstOrderOracle, QuasiNewtonStepInfo[FirstOrderOracle]]
):
    """
    Davidon-Fletcher-Powell (DFP) update for Hessian inverse approximation.

    `H_{k+1} = H_k + (s_k s_k^T) / (y_k^T s_k) - (H_k y_k y_k^T H_k) / (y_k^T H_k y_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[FirstOrderOracle]

    def hess_inv(self, info: QuasiNewtonStepInfo[FirstOrderOracle]) -> Matrix:
        if info.k == 0:
            if info.H is None:
                logger.debug("Initialising Hessian inverse approximation to identity.")
                info.H = Matrix(np.eye(info.x.shape[0], dtype=dtype))
            return info.H
        else:
            logger.debug("Performing DFP Hessian inverse update.")
            H, s, y = info.ensure(info.H), info.ensure(info.s), info.ensure(info.y)
            Hy = Vector(H @ y)
            term1 = Matrix(np.outer(s, s) / Scalar(y.T @ s))
            term2 = Matrix(np.outer(Hy, Hy) / Scalar(y.T @ Hy))
            return Matrix(H + term1 - term2)


class BFGSUpdate(
    QuasiNewtonOptimiser[FirstOrderOracle, QuasiNewtonStepInfo[FirstOrderOracle]]
):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) update for Hessian inverse approximation.

    `H_{k+1} = (I - (s_k y_k^T) / (y_k^T s_k)) H_k (I - (y_k s_k^T) / (y_k^T s_k)) + (s_k s_k^T) / (y_k^T s_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[FirstOrderOracle]

    def hess_inv(self, info: QuasiNewtonStepInfo[FirstOrderOracle]) -> Matrix:
        if info.k == 0:
            if info.H is None:
                logger.debug("Initialising Hessian inverse approximation to identity.")
                info.H = Matrix(np.eye(info.x.shape[0], dtype=dtype))
            return info.H
        else:
            logger.debug("Performing BFGS Hessian inverse update.")
            H, s, y = info.ensure(info.H), info.ensure(info.s), info.ensure(info.y)
            rho = 1 / Scalar(y.T @ s)
            eye = Matrix(np.eye(H.shape[0], dtype=dtype))
            term1 = Matrix(eye - rho * np.outer(s, y))
            term2 = Matrix(eye - rho * np.outer(y, s))
            return Matrix(term1 @ H @ term2 + rho * np.outer(s, s))
