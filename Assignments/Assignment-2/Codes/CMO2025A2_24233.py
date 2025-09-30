# ---------- CMO 2025 Assignment 2 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable, List, Literal, Tuple, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator

sys.path.insert(0, os.path.abspath("oracle_CMO2025A2_py310"))
from oracle_CMO2025A2_py310.oracle_final_CMOA2 import f2, f5  # type: ignore

# Type aliases
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

# Oracle signatures
f2: Callable[[int, bool], Tuple[Matrix, Vector]]
f5: Callable[[int], Tuple[LinearOperator, Vector]]


# ---------- Setup ----------
SRN: int = 24233
"""The 5-digit Student Registration Number (SRN) for the assignment."""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."


# ---------- Implementations ----------
def CD_SOLVE(
    A: Matrix,
    b: Vector,
    x0: Vector | None = None,
    maxiter: int = 100,
) -> Tuple[Vector, List[float], List[float], List[float]]:
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
        alphas (List[float]): List of step sizes `alpha_k`.\\
        numerators (List[float]): List of values `-grad_f(x_k)' u_k`.\\
        lambdas (List[float]): Corresponding eigenvalues `lambda_k`.
    """

    # Initialise x0
    dim: int = b.shape[0]
    if x0 is None:
        x0 = np.zeros_like(b)

    alphas: List[float] = []
    numerators: List[float] = []
    lambdas: List[float] = []

    x: Vector = x0.copy()
    r: Vector = b - A @ x  # r_k = -grad_f(x_k)
    for k in range(min(maxiter, dim)):
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


@overload
def CG_SOLVE(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: Literal[False],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: bool = False,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> Union[
    Tuple[Vector, int, List[float]],
    Tuple[Vector, int, List[float], List[Vector], List[Vector]],
]:
    """
    Standard Conjugate Gradient Method.

    Parameters:
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.

    Returns:
        x (NDArray): Approximate solution vector.\\
        iters (int): Number of iterations taken.\\
        residuals (List[float]): Residual norms `||r_k||_2` at each iteration.

        In addition, if log_directions is set to True, then also return\\
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.\\
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """

    residuals: List[float] = []
    residual_list: List[Vector] = []
    directions: List[Vector] = []

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
        r_normsq: float = float(r @ r)
        alpha_k: float = r_normsq / float(p @ Ap)
        x += alpha_k * p

        r_new: Vector = r - alpha_k * Ap
        beta_k: float = float(r_new @ r_new) / r_normsq
        p = r_new + beta_k * p

        residual_norm: float = float(np.linalg.norm(r_new))
        residuals.append(residual_norm)
        if log_directions:
            residual_list.append(r_new.copy())
            directions.append(p.copy())

        if residual_norm < tol:
            break

        r = r_new

    if log_directions:
        return x, k + 1, residuals, residual_list, directions
    else:
        return x, k + 1, residuals


def GS_ORTHOGONALISE(P: List[Vector], Q: Matrix) -> List[Vector]:
    """
    Gram-Schmidt orthogonalisation.

    Parameters:
        P (List[Vector]): A list (or array) of vectors {p_0,...,p_(m-1)} to be orthogonalised.
        Q (NDArray): SPD matrix (here use `A` from oracle).

    Returns:
        D (List[Vector]): The Q-orthogonalised vectors {d_0,...,d_(m-1)}.
    """

    m: int = len(P)
    D: List[Vector] = []

    for i in range(m):
        p_i: Vector = P[i].copy()
        for j in range(i):
            d_j: Vector = D[j]
            coeff: float = float((p_i @ (Q @ d_j)) / (d_j @ (Q @ d_j)))
            p_i -= coeff * d_j
        D.append(p_i)

    return D


@overload
def CG_SOLVE_FAST(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: Literal[False],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE_FAST(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE_FAST(
    A: Matrix | LinearOperator,
    b: Vector,
    log_directions: bool = False,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> Union[
    Tuple[Vector, int, List[float]],
    Tuple[Vector, int, List[float], List[Vector], List[Vector]],
]:
    """
    Improved Conjugate Gradient Method.

    Parameters:
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.

    Returns:
        x (NDArray): Approximate solution vector.
        iters (int): Number of iterations taken.
        residuals (List[float]): Residual norms `||r_k||_2` at each iteration.

        In addition, if log_directions is set to True, then
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """

    residuals: List[float] = []
    residual_list: List[Vector] = []
    directions: List[Vector] = []

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
        r_normsq: float = float(r @ r)
        alpha_k: float = r_normsq / float(p @ Ap)
        x += alpha_k * p

        r_new: Vector = r - alpha_k * Ap
        beta_k: float = float(r_new @ r_new) / r_normsq
        p = r_new + beta_k * p

        residual_norm: float = float(np.linalg.norm(r_new))
        residuals.append(residual_norm)
        if log_directions:
            residual_list.append(r_new.copy())
            directions.append(p.copy())

        if residual_norm < tol:
            break

        r = r_new

    if log_directions:
        return x, k + 1, residuals, residual_list, directions
    else:
        return x, k + 1, residuals


def NEWTON_SOLVE(
    f_grad: Callable[[Vector], Vector],
    f_hess: Callable[[Vector], Matrix],
    x0: Vector,
    tol: float = 1e-8,
    maxiter: int = 100,
) -> Tuple[Vector, int, List[Vector]]:
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


def SEPARATE_HYPERPLANE() -> Tuple[Vector, float, Tuple[Vector, Vector]]:
    """
    Separating hyperplane (geometry / classification).

    Returns:
        n (NDArray): Normal vector of hyperplane (NumPy array of length 2).
        c (float): Offset (scalar) so that hyperplane is {x: n'x = c}.
        a_closest, b_closest (tuple[NDArray, NDArray]): The closest points in `C_A` and `C_B` used to construct the hyperplane.
    """


def CHECK_FARKAS() -> Tuple[bool]:
    """
    Farkas lemma / infeasibility check.

    Returns:
        feasible: boolean flag (True if feasible).

        If infeasible:
        y_cert: (NDArray) a Farkas certificate satisfying `y >= 0`, `A'y = 0` (numerically), and `b'y < 0` (numerically).

        Diagnostic info (objective value, solver status).
    """
    pass


def rosenbrock(x: Vector) -> float:
    """
    The Rosenbrock function.\\
    `f(x) = (a - x_1)^2 + b (x_2 - x_1^2)^2`

    Parameters:
        x (NDArray): Input point (NumPy array of length 2).

    Returns:
        f (float): Function value.
    """
    a, b = 1, 100
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: Vector) -> Vector:
    """
    Gradient of the Rosenbrock function.\\
    `grad_f(x) = [  -2(a - x_1) - 4b x_1(x_2 - x_1^2),   2b(x_2 - x_1^2) ]'`
    """
    a, b = 1, 100
    grad = np.zeros_like(x)
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    grad[1] = 2 * b * (x[1] - x[0] ** 2)
    return grad


def rosenbrock_hess(x: Vector) -> Matrix:
    """
    Hessian of the Rosenbrock function.\\
    `hess_f(x) = [
        [   2 - 4b(x_2 - 3x_1^2),  -4b x_1 ],
        [   -4b x_1,              2b      ]
    ]`
    """
    _a, b = 1, 100
    hess = np.zeros((2, 2))
    hess[0, 0] = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
    hess[0, 1] = -4 * b * x[0]
    hess[1, 0] = hess[0, 1]  # Symmetric
    hess[1, 1] = 2 * b
    return hess


# ---------- Questions ----------
def question_1():
    A, b = f2(SRN, True)

    # Conjugate Descent
    x_cd, alphas, nums, lambdas = CD_SOLVE(A, b)

    # Conjugate Gradient with logging
    x, iters, residuals, r_list, p_list = CG_SOLVE(A, b, log_directions=True)

    # Gram-Schmidt orthogonalisation
    D = GS_ORTHOGONALISE(p_list, A)


def question_2():
    A, b = f5(SRN)


def question_3():
    pass


def question_4():
    pass


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()
    question_4()

    plt.show()
