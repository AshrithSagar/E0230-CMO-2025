# ---------- CMO 2025 Assignment 2 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable, List, Literal, Optional, Tuple, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LogNorm
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

SAVE_FIGS: bool = True
"""Boolean flag to save the generated plots as PNG files."""

FIGS_DIR: str = "figures"
"""Directory to save the generated plots."""
if SAVE_FIGS and not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)


# ---------- Implementations ----------
def CD_SOLVE(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    maxiter: int = 100,
) -> Tuple[Vector, List[float], List[float], List[float]]:
    """
    Conjugate Direction Method.

    `f(x) = 0.5 x^T A x - b^T x`\\
    `x^* = argmin_x f(x) => A x^* = b`

    Parameters
    -------
        A (NDArray): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        x0 (NDArray, optional): Initial point. Defaults to zero vector.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns
    -------
        x (NDArray): Final iterate after Conjugate Direction.
        alphas (List[float]): List of step sizes `alpha_k`.
        numerators (List[float]): List of values `-grad_f(x_k)^T u_k`.
        lambdas (List[float]): Corresponding eigenvalues `lambda_k`.
    """

    dim: int = b.shape[0]

    # Initialise `x0` to zero vector if not provided
    if x0 is None:
        x0 = np.zeros_like(b)

    # Since `A` is SPD, use `np.linalg.eigh`
    # The eigenpairs are sorted in ascending order of eigenvalues
    eigvals, eigvecs = np.linalg.eigh(A)

    alphas: List[float] = []
    numerators: List[float] = []
    lambdas: List[float] = []

    x: Vector = x0.copy()

    # CD converges in atmost `dim` steps
    k: int = 0
    for k in range(min(maxiter, dim)):
        u: Vector = eigvecs[:, k]  # k-th eigenvector
        lambda_k: float = eigvals[k]  # k-th eigenvalue

        # r_k = -grad_f(x_k) = b - A x_k
        r: Vector = b - A @ x

        rTu: float = float(r @ u)
        if rTu < 0:
            # Ensure `alpha > 0`, since `np.linalg.eigh(A)`
            # returns arbitrary sign for eigenvectors
            rTu = -rTu
            u = -u
        uTAu: float = float(u @ (A @ u))
        alpha: float = rTu / uTAu

        x += alpha * u

        # Can verify that uTAu and lambda_k are same
        assert np.allclose(uTAu, lambda_k), "Denominator and eigenvalue mismatch"

        alphas.append(alpha)
        numerators.append(rTu)
        lambdas.append(lambda_k)

    # Verify that the final iterate is a stationary point
    if k == dim - 1:
        assert np.allclose(A @ x - b, np.zeros_like(b)), "grad_f(x_dim) != 0"

    return x, alphas, numerators, lambdas


@overload
def CG_SOLVE(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[False] = ...,
    tol: float = ...,
    maxiter: int = ...,
    use_relative_tol: bool = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
    use_relative_tol: bool = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: bool = False,
    tol: float = 1e-6,
    maxiter: int = 10_000,
    use_relative_tol: bool = False,
) -> Union[
    Tuple[Vector, int, List[float]],
    Tuple[Vector, int, List[float], List[Vector], List[Vector]],
]:
    """
    Standard Conjugate Gradient Method.

    Parameters
    -------
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.
        use_relative_tol (bool, optional): Boolean flag. Defaults to False.
            When set to True, the stopping rule becomes `||r_k|| / ||r_0|| < tol`.

    Returns
    -------
        x (NDArray): Approximate solution vector.
        iters (int): Number of iterations taken.
        residuals (List[float]): Residual norms `||r_k||_2` at each iteration.
        In addition, if log_directions is set to True, then also return
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """

    residuals: List[float] = []  # ||r_k||_2
    residual_list: List[Vector] = []  # r_k = -grad_f(x_k) = b - A x_k
    directions: List[Vector] = []  # p_k

    ## Initialise for k = 0
    k: int = 0
    x: Vector = np.zeros_like(b)  # x_0 -> zero vector

    r: Vector = b.copy()  # r_0 = (b - A x_0) = b
    rTr: float = float(r @ r)  # r_0' r_0
    r0_norm: float = float(np.sqrt(rTr))  # ||r_0||_2

    p: Vector = r.copy()  # p_0 = r_0

    for k in range(maxiter + 1):  # Need +1 to log the 0-th iterate too
        ## Validate k-th iterate
        r_norm: float = float(np.sqrt(rTr))  # ||r_k||_2

        residuals.append(r_norm)
        if log_directions:
            residual_list.append(r.copy())
            directions.append(p.copy())

        # Stopping condition
        # If satisfied, break and return for k-th iterate
        if use_relative_tol:
            if r_norm / r0_norm < tol:
                break
        else:
            if r_norm < tol:
                break
        if k == maxiter:
            break  # Skip computing (maxiter+1)-th iterate

        ## Compute (k+1)-th iterate
        Ap: Vector = np.asarray(A @ p)  # A p_k
        pTAp: float = float(p @ Ap)  # p_k' A p_k
        assert pTAp > 0, "p_k' A p_k <= 0, A not SPD?"
        if pTAp < 1e-15:
            print(f"[WARN] p_k' A p_k too small ({pTAp}). Stopping CG early.")
            break
        alpha: float = rTr / pTAp  # alpha_k = (r_k' r_k) / (p_k' A p_k)
        if alpha < 0:
            print(f"[WARN] alpha_k < 0 ({alpha}). Stopping CG early.")
            break

        x: Vector = x + alpha * p  # x_{k+1} = x_k + alpha_k p_k
        r: Vector = r - alpha * Ap  # r_{k+1} = r_k - alpha_k A p_k

        rTr_new: float = float(r @ r)  # r_{k+1}' r_{k+1}
        beta: float = rTr_new / rTr  # beta_k = (r_{k+1}' r_{k+1}) / (r_k' r_k)
        p: Vector = r + beta * p  # p_{k+1} = r_{k+1} + beta_k p_k

        rTr: float = rTr_new  # Update rTr for next iteration

    # k is the number of completed iterations (0-indexed)
    # => len(residuals) = len(residual_list) = len(directions) = k + 1
    # => x contains the k-th iterate, i.e., x_k
    # => Number of iterations taken = k + 1 (1-indexed)
    if log_directions:
        return x, k + 1, residuals, residual_list, directions
    else:
        return x, k + 1, residuals


def GS_ORTHOGONALISE(P: List[Vector], Q: Matrix) -> List[Vector]:
    """
    Gram-Schmidt orthogonalisation.

    Parameters
    -------
        P (List[Vector]): A list (or array) of vectors {p_0,...,p_(m-1)} to be orthogonalised.
        Q (NDArray): SPD matrix (here use `A` from oracle).

    Returns
    -------
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


def ichol(A: Union[Matrix, LinearOperator]) -> Matrix:
    """
    Incomplete Cholesky factorization `A ~= L L^T`.
    `A` must be SPD (symmetric positive definite).
    """
    dim: int = A.shape[0]
    L: Matrix = np.zeros((dim, dim))

    for i in range(dim):
        # Standard basis vector e_i
        e_i = np.zeros(dim)
        e_i[i] = 1.0

        # i-th column of A
        c_i: Vector = np.asarray(A @ e_i, dtype=np.float64)

        # Diagonal elements
        sum_diag = np.sum(L[i, :i] ** 2)
        diag_i = c_i[i] - sum_diag
        if diag_i <= 0:
            raise np.linalg.LinAlgError(f"Matrix is not SPD at row {i}.")
        L[i, i] = np.sqrt(diag_i)

        # Off-diagonal elements
        for j in range(i + 1, dim):
            if c_i[j] != 0:  # Preserve sparsity
                sum_lower = np.dot(L[j, :i], L[i, :i])
                L[j, i] = (c_i[j] - sum_lower) / L[i, i]

    return L


@overload
def CG_SOLVE_FAST(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[False] = ...,
    tol: float = ...,
    maxiter: int = ...,
    use_relative_tol: bool = ...,
) -> Tuple[Vector, int, List[float]]: ...
@overload
def CG_SOLVE_FAST(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: Literal[True],
    tol: float = ...,
    maxiter: int = ...,
    use_relative_tol: bool = ...,
) -> Tuple[Vector, int, List[float], List[Vector], List[Vector]]: ...


def CG_SOLVE_FAST(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    log_directions: bool = False,
    tol: float = 1e-6,
    maxiter: int = 10_000,
    use_relative_tol: bool = False,
) -> Union[
    Tuple[Vector, int, List[float]],
    Tuple[Vector, int, List[float], List[Vector], List[Vector]],
]:
    """
    Improved Conjugate Gradient Method.

    Parameters
    -------
        A (NDArray or LinearOperator): SPD matrix from oracle.
        b (NDArray): Right-hand side vector.
        log_directions (bool, optional): Boolean flag. Defaults to False.
            When set to True, the function must additionally return the first `m` residuals and search directions.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10_000.
        use_relative_tol (bool, optional): Boolean flag. Defaults to False.
            When set to True, the stopping rule becomes `||r_k|| / ||r_0|| < tol`.

    Returns
    -------
        x (NDArray): Approximate solution vector.
        iters (int): Number of iterations taken.
        residuals (List[float]): Residual norms `||r_k||_2` at each iteration.

        In addition, if log_directions is set to True, then also return
        residual_list (list[NDArray]): First `m` residuals {r_0,...,r_(m-1)}.
        directions (list[NDArray]): First `m` CG search directions {p_0,...,p_(m-1)}.
    """

    residuals: List[float] = []  # ||r_k||_2
    residual_list: List[Vector] = []  # r_k = -grad_f(x_k) = b - A x_k
    directions: List[Vector] = []  # p_k

    # Preconditioner M_inv
    L = ichol(A)

    def preconditioner_solve(x: Vector):
        y = np.linalg.solve(L, x)  # L y = x
        z = np.linalg.solve(L.T, y)  # L.T z = y
        return z

    M_inv = LinearOperator(
        A.shape,
        matvec=preconditioner_solve,  # type: ignore
    )

    ## Initialise for k = 0
    k: int = 0
    x: Vector = np.zeros_like(b)  # x_0 -> zero vector

    r: Vector = b.copy()  # r_0 = (b - A x_0) = b
    r0_norm: float = float(np.linalg.norm(r))  # ||r_0||_2
    z: Vector = np.asarray(M_inv @ r)  # z_0 = M_inv r_0
    rTz: float = float(r @ z)  # r_0' z_0

    p: Vector = z.copy()  # p_0 = z_0

    for k in range(maxiter + 1):  # Need +1 to log the 0-th iterate too
        ## Validate k-th iterate
        r_norm: float = float(np.linalg.norm(r))  # ||r_k||_2

        residuals.append(r_norm)
        if log_directions:
            residual_list.append(r.copy())
            directions.append(p.copy())

        # Stopping condition
        # If satisfied, break and return for k-th iterate
        if use_relative_tol:
            if r_norm / r0_norm < tol:
                break
        else:
            if r_norm < tol:
                break
        if k == maxiter:
            break  # Skip computing (maxiter+1)-th iterate

        ## Compute (k+1)-th iterate
        Ap: Vector = np.asarray(A @ p)  # A p_k
        pTAp: float = float(p @ Ap)  # p_k' A p_k
        assert pTAp > 0, "p_k' A p_k <= 0, A not SPD?"
        if pTAp < 1e-15:
            print(f"[WARN] p_k' A p_k too small ({pTAp}). Stopping CG early.")
            break
        alpha: float = rTz / pTAp  # alpha_k = (r_k' z_k) / (p_k' A p_k)
        if alpha < 0:
            print(f"[WARN] alpha_k < 0 ({alpha}). Stopping CG early.")
            break

        x: Vector = x + alpha * p  # x_{k+1} = x_k + alpha_k p_k
        r: Vector = r - alpha * Ap  # r_{k+1} = r_k - alpha_k A p_k

        z: Vector = np.asarray(M_inv @ r)  # z_{k+1} = M_inv r_{k+1}
        rTz_new: float = float(r @ z)  # r_{k+1}' z_{k+1}
        beta: float = rTz_new / rTz  # beta_k = (r_{k+1}' z_{k+1}) / (r_k' z_k)
        p: Vector = z + beta * p  # p_{k+1} = z_{k+1} + beta_k p_k

        rTz: float = rTz_new  # Update rTz for next iteration

    # k is the number of completed iterations (0-indexed)
    # => len(residuals) = len(residual_list) = len(directions) = k + 1
    # => x contains the k-th iterate, i.e., x_k
    # => Number of iterations taken = k + 1 (1-indexed)
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

    Parameters
    -------
        f_grad (Callable): Gradient function of f(x).
        f_hess (Callable): Hessian function of f(x).
        x0 (Vector): Initial point. (NumPy array of length 2).
        tol (float, optional): Convergence tolerance. Defaults to 1e-8.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns
    -------
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


# ---------- Helpers ----------
def rosenbrock(x: Vector) -> float:
    """
    The Rosenbrock function.

    `f(x) = (a - x_1)^2 + b (x_2 - x_1^2)^2`

    Parameters
    -------
        x (NDArray): Input point (NumPy array of length 2).

    Returns
    -------
        f (float): Function value.
    """
    a, b = 1, 100
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: Vector) -> Vector:
    """
    Gradient of the Rosenbrock function.\\
    `grad_f(x) = [  -2(a - x_1) - 4b x_1(x_2 - x_1^2),   2b(x_2 - x_1^2) ]^T`
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
        [   2 - 4b(x_2 - 3x_1^2),  -4b x_1  ],
        [   -4b x_1,                2b      ]
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
    print("(alpha_k, -grad_f(x_k)^T u_k, lambda_k):")
    for k, (alpha, numerator, lambda_k) in enumerate(zip(alphas, nums, lambdas)):
        print(f"k={k}: ({alpha:.6f}, {numerator:.6f}, {lambda_k:.6f})")

    ## Q1 Part 2
    # Conjugate Gradient with logging
    print("\n\033[4mPart-2\033[0m:")
    x, iters, residuals, r_list, p_list = CG_SOLVE(A, b, log_directions=True)
    m = iters  # Number of directions computed
    print(f"Number of directions computed: m = {m}")
    print("\nCG search directions:")
    for k, pk in enumerate(p_list):
        print(f"p_{k} = {pk}")
    filename = f"plist_{SRN}.txt"
    np.savetxt(filename, p_list)
    print(f"Saved CG search directions to '{filename}'.")

    # Gram-Schmidt orthogonalisation
    D = GS_ORTHOGONALISE(p_list, A)
    print("\nGram-Schmidt Q-orthogonalised vectors:")
    for k, dk in enumerate(D):
        print(f"d_{k} = {dk}")
    filename = f"dlist_{SRN}.txt"
    np.savetxt(filename, D)
    print(f"Saved Q-orthogonalised vectors to '{filename}'.")

    ## Q1 Part 3
    # Constructing Matrix M
    print("\n\033[4mPart-3\033[0m:")
    D_tilde = [dk / np.sqrt(dk @ (A @ dk)) for dk in D]
    M = np.array([[di @ (A @ dj) for dj in D_tilde] for di in D_tilde])
    filename = f"M_{SRN}.txt"
    np.savetxt(filename, M, fmt="%.6e")
    print(f"Saved Matrix M to '{filename}'.")
    print(f"M = {M}")
    print(f"-> Eigenvalues of M: {np.linalg.eigvals(M)}")
    print(f"-> M close to I: {np.allclose(M, np.eye(m), atol=1e-15)}")
    with np.printoptions(suppress=True):
        print(f"=> M \u2248\n{abs(M)}")

    ## Q1 Part 4
    # `A`-inner-product cosine similarities
    print("\n\033[4mPart-4\033[0m:")
    cos_theta_list = [
        float((pk @ (A @ dk)) / (np.sqrt(pk @ (A @ pk)) * np.sqrt(dk @ (A @ dk))))
        for pk, dk in zip(p_list, D)
    ]
    print(f"List of cosine similarities: {cos_theta_list}")
    print(
        f"-> All close to 1: {np.allclose(cos_theta_list, np.ones(len(cos_theta_list)), atol=1e-15)}"
    )
    with np.printoptions(suppress=True):
        print(f"\u2234 List of cosine similarities: \u2248{np.asarray(cos_theta_list)}")


def question_2():
    print("\n\033[1m\033[4mQuestion-2\033[0m:")
    A, b = f5(SRN)

    ## Q2 Part 1
    print("\033[4mPart-1\033[0m:")
    x1, iters1, res1 = CG_SOLVE(A, b, use_relative_tol=True)
    print(f"CG_SOLVE took {iters1} iterations.")

    # Plot of residual norms `||r_k||_2` vs iteration `k`
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(len(res1)), res1, marker="o")
    plt.title("Conjugate Gradient Residual Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Residual Norm $\|r_k\|_2$")
    plt.grid(True)
    plt.tight_layout()
    print("Plot generated for Conjugate Gradient residual norms.")
    if SAVE_FIGS:
        filename = f"Q2-1-CG_residuals_{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))

    ## Q2 Part 2
    print("\n\033[4mPart-2\033[0m:")
    x2, iters2, res2 = CG_SOLVE_FAST(A, b, use_relative_tol=True)
    print(f"CG_SOLVE_FAST took {iters2} iterations.")

    # Comparision plot of residual norms `||r_k||_2` vs iteration `k` between CG_SOLVE and CG_SOLVE_FAST
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(len(res1)), res1, marker="o", label="CG_SOLVE")
    plt.semilogy(range(len(res2)), res2, marker="o", label="CG_SOLVE_FAST")
    plt.title("Conjugate Gradient vs Improved Conjugate Gradient Residual Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Residual Norm $\|r_k\|_2$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("Plot generated for Improved Conjugate Gradient residual norms.")
    if SAVE_FIGS:
        filename = f"Q2-2-CG_vs_CGFAST_residuals_{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))


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
    plt.figure(figsize=(8, 6))
    x_star = np.array([1.0, 1.0])
    for i, trajectory in enumerate(trajectories):
        errors = [np.linalg.norm(xk - x_star) for xk in trajectory]
        plt.semilogy(range(len(errors)), errors, marker="o", label=f"x0={x0s[i]}")
    plt.title("Newton's Method Error Norms")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Error Norm $\|x_k - x^*\|_2$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("Plot generated for Newton's Method error norms.\n")
    if SAVE_FIGS:
        filename = f"Q3-1a-Newton_error_norms_{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))

    # Four separate contour plots with Newton paths
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(np.array([X, Y]))
    levels = np.logspace(-1, 3, 100)
    for i, trajectory in enumerate(trajectories):
        plt.figure(figsize=(8, 8))
        plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap="jet")
        traj = np.array(trajectory)
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            markersize=3,
            color="black",
            label="Newton path",
        )
        plt.plot(1, 1, marker="*", color="red", markersize=15, label=f"x*={x_star}")
        plt.title(f"Newton's Method Path from x0={x0s[i]}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.tight_layout()
        print(f"Plot generated for Newton's Method path for x0={x0s[i]}.")
        if SAVE_FIGS:
            filename = f"Q3-1b-Newton_path_{i + 1}_{SRN}.jpeg"
            plt.savefig(os.path.join(FIGS_DIR, filename))


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()

    if not SAVE_FIGS:
        plt.show()
    plt.close("all")
