# ---------- CMO 2025 Assignment 2 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable, List, Literal, Optional, Tuple, Union, overload

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
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
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[False] = ...,
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE(
    A: Union[Matrix, LinearOperator],
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
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[False] = ...,
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE_FAST(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE_FAST(
    A: Union[Matrix, LinearOperator],
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
    dim: int = b.shape[0]
    x: Vector = np.zeros_like(b)

    # Preconditioner M_inv: ~diag(A)^(-1/2)
    if isinstance(A, np.ndarray):
        diag_A: Vector = np.diag(A)
    else:
        diag_A: Vector = np.empty(dim)
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            diag_A[i] = np.asarray(A @ e_i)[i]
    diag_A: Vector = np.maximum(diag_A, 1e-15)
    M_inv_diag = 1.0 / np.sqrt(diag_A)
    M_inv = LinearOperator(
        shape=(dim, dim),
        dtype=np.float64,
        matvec=lambda v: M_inv_diag * v,  # type: ignore
    )

    r: Vector = b - np.asarray(A @ x)
    z: Vector = np.asarray(M_inv @ r)
    rTz: float = float(r @ z)
    p: Vector = z.copy()

    residuals.append(float(np.linalg.norm(r)))
    if log_directions:
        residual_list.append(r.copy())
        directions.append(p.copy())

    for k in range(maxiter):
        Ap: Vector = np.asarray(A @ p)
        alpha_k: float = rTz / float(p @ Ap)
        rTz_prev: float = rTz
        x: Vector = x + alpha_k * p
        r: Vector = r - alpha_k * Ap

        r_norm: float = float(np.linalg.norm(r))
        residuals.append(r_norm)
        if log_directions:
            residual_list.append(r.copy())
            directions.append(p.copy())
        if r_norm < tol:
            break

        z: Vector = np.asarray(M_inv @ r)
        rTz: float = float(r @ z)
        beta_k: float = rTz / rTz_prev
        p: Vector = z + beta_k * p

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

    x: Vector = x0.astype(np.float64).copy()
    trajectory: List[Vector] = [x.copy()]

    k: int = 0
    for k in range(maxiter):
        grad: Vector = f_grad(x)
        hess: Matrix = f_hess(x)

        try:
            delta_x: Vector = np.asarray(np.linalg.solve(hess, -grad), dtype=np.float64)
        except np.linalg.LinAlgError:  # Fallback
            print("[WARN] Hessian is singular or not invertible. Using pseudo-inverse.")
            delta_x: Vector = np.asarray(-np.linalg.pinv(hess) @ grad, dtype=np.float64)

        x += delta_x
        trajectory.append(x.copy())

        if np.linalg.norm(grad) < tol:
            break

    return x, k + 1, trajectory


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

    direction: Vector = y - center
    distance: float = float(np.linalg.norm(direction))
    if distance <= radius:
        return y.copy()
    else:
        y_proj: Vector = center + (radius / distance) * direction
        return y_proj


def PROJ_BOX(
    y: Vector,
    low: Vector = np.array([-3.0, 0.0]),
    high: Vector = np.array([3.0, 4.0]),
) -> Vector:
    """
    Projection onto box.

    Parameters:
        y (NDArray): Point to project (NumPy array of length 2).
        low (NDArray, optional): Lower corner of box. Defaults to np.array([-3.0, 0.0]).
        high (NDArray, optional): Upper corner of box. Defaults to np.array([3.0, 4.0]).

    Returns:
        y_proj (NDArray): Projection of `y` on the box (NumPy array of length 2).
    """

    y_proj: Vector = np.minimum(np.maximum(y, low), high)
    return y_proj


def SEPARATE_HYPERPLANE() -> Tuple[Vector, float, Tuple[Vector, Vector]]:
    """
    Separating hyperplane (geometry / classification).

    Returns:
        n (NDArray): Normal vector of hyperplane (NumPy array of length 2).
        c (float): Offset (scalar) so that hyperplane is {x: n'x = c}.
        a_closest, b_closest (tuple[NDArray, NDArray]): The closest points in `C_A` and `C_B` used to construct the hyperplane.
    """

    a_closest = np.array([1.0, 0.0])
    b_closest = np.array([3.0, 0.0])

    # Normal vector
    n = b_closest - a_closest
    n = n / np.linalg.norm(n)

    # Offset
    m: Vector = (a_closest + b_closest) / 2  # Midpoint
    c: float = float(n @ m)

    return n, c, (a_closest, b_closest)


def CHECK_FARKAS() -> Tuple[bool, Optional[Vector], dict]:
    """
    Farkas lemma / infeasibility check.

    Returns:
        feasible: boolean flag (True if feasible).

        If infeasible:
        y_cert: (NDArray) a Farkas certificate satisfying `y >= 0`, `A'y = 0` (numerically), and `b'y < 0` (numerically).

        Diagnostic info (objective value, solver status).
    """
    A: Matrix = np.array([[1, 1], [-1, 0], [0, -1]], dtype=np.float64)
    b: Vector = np.array([-1, 0, 0], dtype=np.float64)

    # Primal feasibility problem
    x = cp.Variable(2)
    constraints: List[cp.Constraint] = [A[i] @ x <= b[i] for i in range(len(b))]
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if prob.status in ["infeasible", "infeasible_inaccurate"]:
        # Extract dual multipliers
        y_vals: List[float] = []
        for c in constraints:
            if c.dual_value is not None:
                val = c.dual_value
                if isinstance(val, (np.ndarray, list)):
                    val = val[0]
                if val is not None:
                    y_vals.append(float(val))
                else:
                    y_vals.append(0.0)  # Fallback
            else:
                y_vals.append(0.0)  # Fallback
        y: Vector = np.array(y_vals, dtype=np.float64)
        y: Vector = np.maximum(y, 0)  # Ensure nonnegativity
        # Normalize certificate
        if np.linalg.norm(A.T @ y) < 1e-6 and b @ y < -1e-6:
            return False, y, {"status": prob.status, "bTy": b @ y, "ATy": A.T @ y}
        else:
            return False, None, {"status": prob.status, "dual_failed": True}
    else:
        return True, None, {"status": prob.status}


# ---------- Helpers ----------
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
    print("\n\033[1m\033[4mQuestion-1\033[0m:")
    A, b = f2(SRN, True)

    ## Q1 Part 1
    # Conjugate Descent
    print("\033[4mPart-1\033[0m:")
    x_cd, alphas, nums, lambdas = CD_SOLVE(A, b)
    print("(alpha_k, -grad_f(x_k)' @ u_k, lambda_k) for first 7 iterations:")
    for k in range(8):
        print(f"k={k}: ({alphas[k]:.6f}, {nums[k]:.6f}, {lambdas[k]:.6f})")

    ## Q1 Part 2
    # Conjugate Gradient with logging
    print("\n\033[4mPart-2\033[0m:")
    x, iters, residuals, r_list, p_list = CG_SOLVE(A, b, log_directions=True)
    print(f"Number of directions computed: m = {iters + 1}")
    print("\nCG search directions:")
    for i, pk in enumerate(p_list):
        print(f"p_{i} = {pk}")

    # Gram-Schmidt orthogonalisation
    D = GS_ORTHOGONALISE(p_list, A)
    print("\nGram-Schmidt Q-orthogonalised vectors:")
    for i, dk in enumerate(D):
        print(f"d_{i} = {dk}")

    ## Q1 Part 3
    # Constructing Matrix M
    print("\n\033[4mPart-3\033[0m:")
    D_tilde: List[Vector] = []
    for k in range(len(D)):
        dk: Vector = D[k]
        dk_tilde: Vector = dk / np.sqrt(dk @ (A @ dk))
        D_tilde.append(dk_tilde)
    M = np.zeros((len(D_tilde), len(D_tilde)))
    for i in range(len(D_tilde)):
        for j in range(len(D_tilde)):
            di_tilde = D_tilde[i]
            dj_tilde = D_tilde[j]
            M[i, j] = float(di_tilde @ (A @ dj_tilde))
    print(f"M = {M}")
    print(f"Eigenvalues of M: {np.linalg.eigvals(M)}")
    print(f"M close to I: {np.allclose(M, np.eye(len(D_tilde)), atol=1e-15)}")

    ## Q1 Part 4
    # `A`-inner-product cosine similarities
    print("\n\033[4mPart-4\033[0m:")
    cos_theta_list: List[float] = []
    for k in range(len(p_list)):
        pk: Vector = p_list[k]
        dk: Vector = D[k]
        q_cosine: float = float(
            (pk @ (A @ dk)) / (np.sqrt(pk @ (A @ pk)) * np.sqrt(dk @ (A @ dk)))
        )
        cos_theta_list.append(q_cosine)
    print(f"List of cosine similarities: {cos_theta_list}")


def question_2():
    print("\n\033[1m\033[4mQuestion-2\033[0m:")
    A, b = f5(SRN)

    ## Q2 Part 1
    print("\033[4mPart-1\033[0m:")
    x1, iters1, res1 = CG_SOLVE(A, b)
    print(f"CG_SOLVE took {iters1} iterations.")

    # Plot of residual norms `||r_k||_2` vs iteration `k`
    plt.figure()
    plt.semilogy(range(len(res1)), res1, marker="o")
    plt.title("Conjugate Gradient Residual Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Residual Norm $\|r_k\|_2$")
    plt.grid(True)
    print("Plot generated for Conjugate Gradient residual norms.")

    ## Q2 Part 2
    print("\n\033[4mPart-2\033[0m:")
    x2, iters2, res2 = CG_SOLVE_FAST(A, b)
    print(f"CG_SOLVE_FAST took {iters2} iterations.")

    # Comparision plot of residual norms `||r_k||_2` vs iteration `k` between CG_SOLVE and CG_SOLVE_FAST
    plt.figure()
    plt.semilogy(range(len(res1)), res1, marker="o", label="CG_SOLVE")
    plt.semilogy(range(len(res2)), res2, marker="o", label="CG_SOLVE_FAST")
    plt.title("Conjugate Gradient vs Improved Conjugate Gradient Residual Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Residual Norm $\|r_k\|_2$")
    plt.legend()
    plt.grid(True)
    print("Plot generated for Improved Conjugate Gradient residual norms.")


def question_3():
    print("\n\033[1m\033[4mQuestion-3\033[0m:")

    ## Q3 Part 1
    print("\033[4mPart-1\033[0m:")
    x0s: List[Vector] = [
        np.array([2, 2]),
        np.array([5, 5]),
        np.array([-10, -4]),
        np.array([50, 60]),
    ]
    xs: List[Vector] = []
    trajectories: List[List[Vector]] = []
    for x0 in x0s:
        x, iters, trajectory = NEWTON_SOLVE(
            f_grad=rosenbrock_grad, f_hess=rosenbrock_hess, x0=x0
        )
        xs.append(x)
        trajectories.append(trajectory)

    # Plot of error `||x_k - x^*||_2` vs iteration `k` for each starting point
    plt.figure()
    x_star = np.array([1.0, 1.0])
    for i, trajectory in enumerate(trajectories):
        errors = [np.linalg.norm(xk - x_star) for xk in trajectory]
        plt.semilogy(range(len(errors)), errors, marker="o", label=f"x0={x0s[i]}")
    plt.title("Newton's Method Error Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Error Norm $\|x_k - x^*\|_2$")
    plt.legend()
    plt.grid(True)
    print("Plot generated for Newton's Method error norms.\n")

    # Four separate contour plots with Newton paths
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(np.array([X, Y]))
    levels = np.logspace(-1, 3, 100)
    for i, trajectory in enumerate(trajectories):
        plt.figure()
        plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap="jet")
        traj_array = np.array(trajectory)
        plt.plot(traj_array[:, 0], traj_array[:, 1], marker="o", color="black")
        plt.plot(1, 1, marker="*", color="red", markersize=15)  # Global minimum
        plt.title(f"Newton's Method Path from x0={x0s[i]}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        print(f"Plot generated for Newton's Method path for x0={x0s[i]}.")


def question_4():
    print("\n\033[1m\033[4mQuestion-4\033[0m:")

    ## Q4 Part 1
    # Projections in a navigation problem
    print("\033[4mPart-1\033[0m:")
    points: List[Vector] = [
        np.array([6.0, 6.0]),
        np.array([2.0, 3.0]),
        np.array([-4.0, -1.0]),
    ]

    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 7))
    circle = Circle((0, 0), 5, color="lightblue", alpha=0.5)
    ax.add_artist(circle)
    ax.add_patch(Rectangle((-3, 0), 6, 4, color="orange", alpha=0.5))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_title(r"Projections onto Circle ($C_1$) and Box ($C_2$)")
    for p in points:
        for proj in [PROJ_CIRCLE(p), PROJ_BOX(p)]:
            ax.plot(*p, "ro")
            ax.plot(*proj, "go")
            ax.arrow(*p, *(proj - p), head_width=0.2, color="gray", linestyle="--")
    ax.legend([r"Circle ($C_1$)", r"Box ($C_2$)", "Original point", "Projected point"])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.grid(True)
    plt.tight_layout()
    print("Plot generated for projections onto circle and box.")

    ## Q4 Part 2
    # Separating hyperplane in a classification story
    print("\n\033[4mPart-2\033[0m:")
    n, c, (a_closest, b_closest) = SEPARATE_HYPERPLANE()
    fig, ax = plt.subplots(figsize=(12, 6))

    # C_A
    circle = Circle((0, 0), 1, color="blue", alpha=0.3, label=r"Group A ($C_A$)")
    ax.add_artist(circle)

    # C_B
    y = np.linspace(-2, 2, 100)
    ax.fill_betweenx(y, 3, 5, color="red", alpha=0.3, label=r"Group B ($C_B$)")

    # Closest points
    ax.plot(*a_closest, "bo", label="Closest point in C_A")
    ax.plot(*b_closest, "ro", label="Closest point in C_B")

    # Separating hyperplane: n^T x = c
    # Solve for x2 given x1: n1 * x1 + n2 * x2 = c => x2 = (c - n1 * x1) / n2
    x_vals = np.linspace(-1.5, 4, 300)
    if abs(n[1]) > 1e-6:
        y_vals = (c - n[0] * x_vals) / n[1]
        ax.plot(x_vals, y_vals, "k--", label="Separating Hyperplane")
    else:
        # vertical line x = c / n[0]
        x_hyper = c / n[0]
        ax.axvline(x=x_hyper, color="k", linestyle="--", label="Separating Hyperplane")

    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("Separating Hyperplane between Group A and Group B")
    ax.grid(True)
    ax.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    print("Plot generated for separating hyperplane between two groups.")

    ## Q4 Part 3
    # Farkas lemma in a supply-chain model
    print("\n\033[4mPart-3\033[0m:")
    feasible, y_cert, info = CHECK_FARKAS()
    print("Is feasible:", feasible)
    print("Certificate y:", y_cert)
    print("Diagnostics:", info)


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()
    question_4()

    plt.show()
