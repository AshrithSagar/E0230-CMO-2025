# ---------- CMO 2025 Assignment 2 ----------

# ---------- Imports ----------
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

# Type aliases
type Vector = NDArray[np.float64]
type Matrix = NDArray[np.float64]


# ---------- Implementations ----------
def CD_SOLVE(
    A: Matrix,
    b: Vector,
    x0: Vector | None = None,
    maxiter: int = 100,
) -> tuple[Vector, list[float], list[float], list[float]]:
    """
    Conjugate Direction Method.

    `f(x) = 0.5 x'Ax - b'x`\\
    `x^* = argmin_x f(x) => A x^* = b`

    Parameters:
        A (NDArray): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        x0 (NDArray, optional): Initial point. Defaults to zero vector.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        x (NDArray): Final iterate after Conjugate Descent.\\
        alphas (list[float]): List of step sizes `alpha_k`.\\
        numerators (list[float]): List of values `-grad_f(x_k)' u_k`.\\
        lambdas (list[float]): Corresponding eigenvalues `lambda_k`.
    """

    # Initialise x0
    if x0 is None:
        x0 = np.zeros_like(b)

    alphas: list[float] = []
    numerators: list[float] = []
    lambdas: list[float] = []

    x: Vector = x0.copy()
    r: Vector = b - A @ x  # r_k = -grad_f(x_k)
    for k in range(maxiter):
        # Generate u_k = e_k for initial search direction
        u = np.zeros_like(b)
        u[k] = 1.0

        Au: Vector = A @ u
        numerator: float = float(r @ u)
        lambda_k: float = float(u @ Au)
        alpha_k: float = numerator / lambda_k

        x += alpha_k * u
        r -= alpha_k * Au

        alphas.append(alpha_k)
        numerators.append(-numerator)
        lambdas.append(lambda_k)

    return x, alphas, numerators, lambdas


def CG_SOLVE(
    A: Matrix | LinearOperator,
    b: Vector,
    tol: float = 1e-6,
    maxiter: int = 10_000,
    log_directions: bool = False,
) -> (
    tuple[Vector, int, list[float]]
    | tuple[Vector, int, list[float], list[Vector], list[Vector]]
):
    """
    Standard Conjugate Gradient Method.

    Parameters:
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.

    Returns:
        x (NDArray): Approximate solution vector.\\
        iters (int): Number of iterations taken.\\
        residuals (list[float]): Residual norms `||r_k||_2` at each iteration.

        In addition, if log_directions is set to True, then also return\\
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.\\
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """

    residuals: list[float] = []
    residual_list: list[Vector] = []
    directions: list[Vector] = []

    # Initialise x0
    k: int = 0
    x: Vector = np.zeros_like(b)

    r: Vector = b - np.asarray(A @ x)
    p: Vector = r.copy()

    residuals.append(float(np.linalg.norm(r)))
    if log_directions:
        residual_list.append(r.copy())
        directions.append(p.copy())

    for k in range(maxiter):
        Ap: Vector = np.asarray(A @ p)
        alpha_k: float = float(r @ r) / float(p @ Ap)

        x += alpha_k * p
        r_new: Vector = r - alpha_k * Ap

        residual_norm: float = float(np.linalg.norm(r_new))
        residuals.append(residual_norm)
        if log_directions and k + 1 < maxiter:
            residual_list.append(r_new.copy())

        if residual_norm < tol:
            break

        beta_k: float = float(r_new @ r_new) / float(r @ r)
        p = r_new + beta_k * p

        if log_directions and k + 1 < maxiter:
            directions.append(p.copy())

        r = r_new

    if log_directions:
        return x, k + 1, residuals, residual_list, directions
    else:
        return x, k + 1, residuals


def GS_ORTHOGONALISE(P: list[Vector], Q: Matrix) -> list[Vector]:
    """
    Gram-Schmidt orthogonalisation.

    Parameters:
        P (list[Vector]): A list (or array) of vectors {p_0,...,p_(m-1)} to be orthogonalised.
        Q (NDArray): SPD matrix (here use `A` from oracle).

    Returns:
        D (list[Vector]): The Q-orthogonalised vectors {d_0,...,d_(m-1)}.
    """


def CG_SOLVE_FAST(
    A: Matrix | LinearOperator,
    b: Vector,
    tol: float = 1e-6,
    maxiter: int = 10_000,
    log_directions: bool = False,
) -> (
    tuple[Vector, int, list[float]]
    | tuple[Vector, int, list[float], list[Vector], list[Vector]]
):
    """
    Improved Conjugate Gradient Method.

    Parameters:
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.

    Returns:
        x (NDArray): Approximate solution vector.
        iters (int): Number of iterations taken.
        residuals (list[float]): Residual norms `||r_k||_2` at each iteration.

        In addition, if log_directions is set to True, then
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """


def NEWTON_SOLVE(
    f_grad: Callable[[Vector], Vector],
    f_hess: Callable[[Vector], Vector],
    x0: Vector,
    tol: float = 1e-8,
    maxiter: int = 100,
) -> tuple[Vector, int, list[Vector]]:
    """
    Newton's Method.

    Parameters:
        f_grad (Callable): Gradient function of f(x).
        f_hess (Callable): Hessian function of f(x).
        x0 (Vector): Initial point. (NumPy array of length 2).
        tol (float, optional): Convergence tolerance. Defaults to 1e-8.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        x (NDArray): Final iterate.
        iters (int): Number of iterations performed.
        trajectory (list[NDArray]): List of iterates (for plotting Newton paths).
    """


def PROJ_CIRCLE(
    y: Vector,
    center: Vector = np.array([0.0, 0.0]),
    radius: float = 5.0,
) -> Vector:
    """
    Projection onto circle.

    Parameters:
        y (NDArray): Point to project (NumPy array of length 2).
        center (NDArray, optional): Centre of circle. Defaults to np.array([0.0, 0.0]).
        radius (float, optional): Radius of circle. Defaults to 5.0.

    Returns:
        y_proj (NDArray): Projection of `y` on the closed Euclidean ball (NumPy array of length 2).
    """


def PROJ_BOX(
    y: Vector,
    center: Vector = np.array([0.0, 0.0]),
    radius: float = 5.0,
) -> Vector:
    """
    Projection onto box.

    Parameters:
        y (NDArray): Point to project (NumPy array of length 2).
        low (NDArray, optional): Lower corner of box. Defaults to np.array([-3.0, 0.0]).
        high (NDArray, optional): Upper corner of box. Defaults to np.array([3.0, 4.0]).

    Returns:
        y_proj (NDArray): Projection of `y` on the closed Euclidean ball (NumPy array of length 2).
    """


def SEPARATE_HYPERPLANE() -> tuple[Vector, float, tuple[Vector, Vector]]:
    """
    Separating hyperplane (geometry / classification).

    Returns:
        n (NDArray): Normal vector of hyperplane (NumPy array of length 2).
        c (float): Offset (scalar) so that hyperplane is {x: n'x = c}.
        a_closest, b_closest (tuple[NDArray, NDArray]): The closest points in `C_A` and `C_B` used to construct the hyperplane.
    """


def CHECK_FARKAS() -> tuple[bool]:
    """
    Farkas lemma / infeasibility check.

    Returns:
        feasible: boolean flag (True if feasible).

        If infeasible:
        y_cert: (NDArray) a Farkas certificate satisfying `y >= 0`, `A'y = 0` (numerically), and `b'y < 0` (numerically).

        Diagnostic info (objective value, solver status).
    """
    pass
