"""
Newton and Quasi-Newton methods
=======
src/cmo/unconstrained/quasi_newton.py
"""

import numpy as np

from cmo.core.base import (
    NewtonDirectionMixin,
    QuasiNewtonOptimiser,
    UnitStepLengthMixin,
)
from cmo.core.info import QuasiNewtonStepInfo, SecondOrderLineSearchStepInfo
from cmo.core.oracle import FirstOrderOracle, SecondOrderOracle
from cmo.functions.protocol import FirstOrderFunctionProto, SecondOrderFunctionProto
from cmo.types import Matrix, Scalar, Vector, dtype
from cmo.utils.logging import logger


class NewtonMethod(
    NewtonDirectionMixin[
        SecondOrderFunctionProto,
        SecondOrderOracle[SecondOrderFunctionProto],
        SecondOrderLineSearchStepInfo[
            SecondOrderFunctionProto, SecondOrderOracle[SecondOrderFunctionProto]
        ],
    ],
    UnitStepLengthMixin[
        SecondOrderFunctionProto,
        SecondOrderOracle[SecondOrderFunctionProto],
        SecondOrderLineSearchStepInfo[
            SecondOrderFunctionProto, SecondOrderOracle[SecondOrderFunctionProto]
        ],
    ],
):
    """
    Standard Newton's method.

    `x_{k+1} = x_k - [f''(x_k)]^{-1} f'(x_k)`
    """

    StepInfoClass = SecondOrderLineSearchStepInfo[
        SecondOrderFunctionProto, SecondOrderOracle[SecondOrderFunctionProto]
    ]

    pass  # All methods are provided by the mixins


class SR1Update(
    QuasiNewtonOptimiser[
        FirstOrderFunctionProto,
        FirstOrderOracle[FirstOrderFunctionProto],
        QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ]
):
    """
    Symmetric Rank-One (SR1) update for Hessian inverse approximation.

    `H_{k+1} = H_k + ((s_k - H_k y_k)(s_k - H_k y_k)^T) / ((s_k - H_k y_k)^T y_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[
        FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
    ]

    def hess_inv(
        self,
        info: QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ) -> Matrix:
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
    QuasiNewtonOptimiser[
        FirstOrderFunctionProto,
        FirstOrderOracle[FirstOrderFunctionProto],
        QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ]
):
    """
    Davidon-Fletcher-Powell (DFP) update for Hessian inverse approximation.

    `H_{k+1} = H_k + (s_k s_k^T) / (y_k^T s_k) - (H_k y_k y_k^T H_k) / (y_k^T H_k y_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[
        FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
    ]

    def hess_inv(
        self,
        info: QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ) -> Matrix:
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
    QuasiNewtonOptimiser[
        FirstOrderFunctionProto,
        FirstOrderOracle[FirstOrderFunctionProto],
        QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ]
):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) update for Hessian inverse approximation.

    `H_{k+1} = (I - (s_k y_k^T) / (y_k^T s_k)) H_k (I - (y_k s_k^T) / (y_k^T s_k)) + (s_k s_k^T) / (y_k^T s_k)`
    """

    StepInfoClass = QuasiNewtonStepInfo[
        FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
    ]

    def hess_inv(
        self,
        info: QuasiNewtonStepInfo[
            FirstOrderFunctionProto, FirstOrderOracle[FirstOrderFunctionProto]
        ],
    ) -> Matrix:
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
