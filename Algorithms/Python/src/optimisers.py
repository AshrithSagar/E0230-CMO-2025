"""
src/optimisers.py
=======
Optimisation algorithms.

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

import numpy as np

from .base import (
    ExactLineSearchMixin,
    LineSearchOptimiser,
    NewtonDirectionMixin,
    SteepestDescentDirectionMixin,
)
from .functions import ConvexQuadratic
from .info import FirstOrderLineSearchStepInfo, SecondOrderLineSearchStepInfo
from .types import floatVec


class GradientDescent(SteepestDescentDirectionMixin):
    """
    Standard gradient descent.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`
    """

    def reset(self):
        self.lr = float(self.config.get("lr", 1e-3))
        return super().reset()

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        return self.lr


class GradientDescentExactLineSearch(
    SteepestDescentDirectionMixin, ExactLineSearchMixin
):
    """
    Gradient descent with exact line search for convex quadratic functions.

    `x_{k+1} = x_k - alpha_k * f'(x_k)`\\
    where `alpha_k = (f'(x_k)^T f'(x_k)) / (f'(x_k)^T Q f'(x_k))`
    """

    pass  # All methods are provided by the mixins


class GradientDescentArmijo(SteepestDescentDirectionMixin):
    """
    Forward-expansion Armijo line search:\\
    Increase alpha until Armijo condition holds (or until safe cap).

    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k`
    """

    def reset(self):
        self.c = float(self.config.get("c", 1e-4))  # Armijo parameter
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_start = float(self.config.get("alpha_start", 0.0))
        self.alpha_step = float(self.config.get("alpha_step", 1e-1))
        self.alpha_stop = float(self.config.get("alpha_stop", 1.0))

        assert 0 < self.c < 1, "c must be in (0, 1)"
        return super().reset()

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        if info.direction is None:
            raise ValueError("Direction must be provided for line search.")
        d = info.direction
        f = info.fx
        grad = info.dfx
        derphi0 = float(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # Forward expansion
        alpha_prev: float | None = None
        for alpha in np.arange(
            self.alpha_start,
            self.alpha_stop + self.alpha_step,
            self.alpha_step,
            dtype=np.float64,
        ):
            f_new, _ = self._phi_and_derphi(info, alpha)
            if f_new <= f + self.c * alpha * derphi0:
                alpha_prev = alpha
            else:
                break
        if alpha_prev is not None:
            return alpha_prev
        else:
            return self.alpha_min


class GradientDescentBacktracking(SteepestDescentDirectionMixin):
    """
    Standard backtracking Armijo (decreasing alpha).
    """

    def reset(self):
        self.c = float(self.config.get("c", 1e-4))  # Armijo parameter
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c < 1, "c must be in (0, 1)"
        return super().reset()

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        if info.direction is None:
            raise ValueError("Direction must be provided for line search.")
        d = info.direction
        f = info.fx
        grad = info.dfx
        derphi0 = float(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        alpha = self.alpha_init
        for _ in range(self.maxiter):
            new_f, _ = self._phi_and_derphi(info, alpha)
            if new_f <= f + self.c * alpha * derphi0:
                return alpha
            alpha *= self.beta
            if alpha < self.alpha_min:
                return self.alpha_min
        return alpha


class GradientDescentArmijoGoldstein(SteepestDescentDirectionMixin):
    """
    Armijo-Goldstein via expansion to bracket and then bisection.\\
    `f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha_k * f'(x_k)^T p_k` (Armijo)\\
    `f(x_k + alpha_k * p_k) >= f(x_k) + (1 - c) * alpha_k * f'(x_k)^T p_k` (Goldstein)
    """

    def reset(self):
        self.c = float(self.config.get("c", 1e-4))  # Armijo-Goldstein parameter
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c < 0.5, "c must be in (0, 0.5)"
        return super().reset()

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        if info.direction is None:
            raise ValueError("Direction must be provided for line search.")
        d = info.direction
        f = info.fx
        grad = info.dfx
        derphi0 = float(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        # If initial alpha already satisfies both, return it
        alpha = self.alpha_init
        f_new, _ = self._phi_and_derphi(info, alpha)
        if (f_new <= f + self.c * alpha * derphi0) and (
            f_new >= f + (1 - self.c) * alpha * derphi0
        ):
            return alpha

        # Expand to find an interval [alpha_lo, alpha_hi] where Armijo condition is satisfied at alpha_hi
        alpha_lo = 0.0
        alpha_hi = alpha
        for _ in range(self.maxiter):
            phi_hi, _ = self._phi_and_derphi(info, alpha_hi)
            if phi_hi <= f + self.c * alpha_hi * derphi0:
                break
            alpha_hi *= self.beta
            if alpha_hi > self.alpha_max:
                break

        # Bisect between alpha_lo and alpha_hi until Goldstein condition hold
        for _ in range(self.maxiter):
            alpha_mid = 0.5 * (alpha_lo + alpha_hi)
            phi_mid, _ = self._phi_and_derphi(info, alpha_mid)
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


class GradientDescentWolfe(SteepestDescentDirectionMixin):
    """
    Strong Wolfe line search using bracket + zoom (Nocedal & Wright).\\
    `phi(alpha_k) <= phi(0) + c1 * alpha_k * phi'(0)` (Armijo)\\
    `|phi'(alpha_k)| <= c2 * |phi'(0)|` (Strong curvature)\\
    where `phi(alpha_k) = f(x_k + alpha_k * p_k)`, `phi'(alpha_k) = f'(x_k + alpha_k * p_k)^T p_k`.
    """

    def reset(self):
        self.c1 = float(self.config.get("c1", 1e-4))
        self.c2 = float(self.config.get("c2", 0.9))
        self.beta = float(self.config.get("beta", 0.5))
        self.alpha_init = float(self.config.get("alpha_init", 1.0))
        self.alpha_min = float(self.config.get("alpha_min", 1e-14))
        self.alpha_max = float(self.config.get("alpha_max", 1e6))
        self.maxiter = int(self.config.get("maxiter", 10))

        assert 0 < self.c1 < self.c2 < 1, "0 < c1 < c2 < 1 must be satisfied"
        return super().reset()

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        if info.direction is None:
            raise ValueError("Direction must be provided for line search.")
        d = info.direction
        f = info.fx
        grad = info.dfx
        derphi0 = float(grad.T @ d)
        # Fallback if directional derivative is non-negative
        if derphi0 >= 0:
            return self.alpha_min

        phi0 = f
        alpha = self.alpha_init
        phi_prev = phi0
        alpha_prev = 0.0

        for i in range(self.maxiter):
            phi_a, derphi_a = self._phi_and_derphi(info, alpha)

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
        info: FirstOrderLineSearchStepInfo,
        alpha_lo: float,
        alpha_hi: float,
        phi0: float,
        derphi0: float,
        maxiter: int = 50,
    ):
        """
        Zoom procedure as in Nocedal & Wright (uses safe bisection interpolation).
        Returns an alpha that satisfies strong Wolfe (if found), otherwise the best found.
        """
        if info.direction is None:
            raise ValueError("Direction must be provided for line search.")
        phi_lo, _derphi_lo = self._phi_and_derphi(info, alpha_lo)
        for _ in range(maxiter):
            alpha_j = 0.5 * (alpha_lo + alpha_hi)  # safe midpoint
            phi_j, derphi_j = self._phi_and_derphi(info, alpha_j)

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


class ConjugateDirectionMethod(LineSearchOptimiser[FirstOrderLineSearchStepInfo]):
    """
    Linear conjugate direction method for convex quadratic functions.\\
    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    StepInfoClass = FirstOrderLineSearchStepInfo

    def reset(self):
        self.line_search = ExactLineSearchMixin().reset()

        if self.config.get("directions") is None:
            raise ValueError(f"{self.__class__.__name__} requires directions apriori.")
        self.directions: list[floatVec] = self.config.get("directions", [])

        return super().reset()

    def direction(self, info: FirstOrderLineSearchStepInfo) -> floatVec:
        k = info.k
        if k < len(self.directions):
            direction = self.directions[k]
            self.line_search.step_directions.append(direction)
            return direction
        else:
            raise IndexError(f"No more directions available for iteration {k}.")

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        alpha = self.line_search.step_length(info)
        self.step_directions[-1] = self.line_search.step_directions[-1]
        return alpha


class ConjugateGradientMethod(LineSearchOptimiser[FirstOrderLineSearchStepInfo]):
    """
    Linear conjugate gradient method for convex quadratic functions.\\
    `x_{k+1} = x_k + alpha_k * p_k`\\
    where `p_k` are conjugate directions and `alpha_k` is the exact line search step length.
    """

    StepInfoClass = FirstOrderLineSearchStepInfo

    def reset(self):
        self.line_search = ExactLineSearchMixin().reset()
        self.rTr_prev: float
        return super().reset()

    def direction(self, info: FirstOrderLineSearchStepInfo) -> floatVec:
        if not isinstance(info.oracle._oracle_f, ConvexQuadratic):
            raise NotImplementedError(
                f"This implementation of {self.__class__.__name__} requires a ConvexQuadratic Function."
            )

        k = info.k
        grad = info.dfx

        if k == 0:
            self.rTr_prev = float(grad.T @ grad)
            direction = -grad
        else:
            rTr = float(grad.T @ grad)
            beta = rTr / self.rTr_prev
            direction = grad + beta * self.step_directions[-1]
            self.rTr_prev = rTr

        self.line_search.step_directions.append(direction)
        return direction

    def step_length(self, info: FirstOrderLineSearchStepInfo) -> float:
        alpha = self.line_search.step_length(info)
        self.step_directions[-1] = self.line_search.step_directions[-1]
        return alpha


class NewtonMethod(NewtonDirectionMixin):
    """
    Standard Newton's method.

    `x_{k+1} = x_k - [f''(x_k)]^{-1} f'(x_k)`
    """

    def step_length(self, info: SecondOrderLineSearchStepInfo) -> float:
        return 1.0
