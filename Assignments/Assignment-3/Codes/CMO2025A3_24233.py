# ---------- CMO 2025 Assignment 3 ----------

# ---------- Imports ----------
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.abspath("oracle_2025A3"))
from oracle_2025A3 import f1  # type: ignore

# Type aliases
Scalar: TypeAlias = float
"""A type alias for a scalar real number."""

Vector: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 1D numpy array of real numbers."""

Matrix: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 2D numpy array of real numbers."""

# Oracle signatures
f1: Callable[[int], None]


# ---------- Setup ----------
SRN: int = 24233
"""The 5-digit Student Registration Number (SRN) for the assignment."""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."

DUP_COL_IDX: int = 5
"""Index of the feature column to duplicate in Question 1, Part 5."""
assert isinstance(DUP_COL_IDX, int) and 0 <= DUP_COL_IDX < 15, (
    "DUP_COL_IDX must be a valid column index in X."
)

SAVE_FIGS: bool = True
"""Boolean flag to save the generated plots as PNG files."""

FIGS_DIR: str = "figures"
"""Directory to save the generated plots."""
if SAVE_FIGS and not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)


# ---------- Implementations ----------
def LASSO_REGRESSION(X: Matrix, y: Vector, lam: Scalar) -> Vector:
    """
    Solve the Lasso regression problem for the Linear objective function using CVXPY.

    `min_{beta} 0.5 ||X beta - y||_2^2 + lam ||beta||_1`

    Parameters:
        X (Matrix): Feature matrix. Shape (n_samples, n_features).
        y (Vector): Response vector. Shape (n_samples,).
        lam (Scalar): Regularisation parameter. Must be non-negative.

    Returns:
        beta (Vector): Estimated coefficients.
    """

    assert lam >= 0, "Regularisation parameter must be non-negative."

    n_features = X.shape[1]
    beta = cp.Variable(n_features)

    objective = cp.Minimize(0.5 * cp.sum_squares(X @ beta - y) + lam * cp.norm1(beta))
    problem = cp.Problem(objective)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimisation failed with status: {problem.status}")

    return np.array(beta.value, dtype=np.double).flatten()


def LASSO_REGRESSION_DUAL(X: Matrix, y: Vector, lam: Scalar) -> Vector:
    """
    Solve the dual of the Lasso regression problem using CVXPY.

    `max_{u} -0.5 ||u||_2^2 + y^T u`\\
    `subject to ||X^T u||_infty <= lam`

    Parameters:
        X (Matrix): Feature matrix. Shape (n_samples, n_features).
        y (Vector): Response vector. Shape (n_samples,).
        lam (Scalar): Regularisation parameter. Must be non-negative.

    Returns:
        u (Vector): Dual variable.
    """

    assert lam >= 0, "Regularisation parameter must be non-negative."

    n_samples = X.shape[0]
    u = cp.Variable(n_samples)

    objective = cp.Maximize(-0.5 * cp.sum_squares(u) + y @ u)
    constraints: List[cp.Constraint] = [cp.norm_inf(X.T @ u) <= lam]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimisation failed with status: {problem.status}")

    return np.array(u.value, dtype=np.double).flatten()


def PROJ_CIRCLE(
    y: Vector,
    center: Vector = np.array([0.0, 0.0]),
    radius: Scalar = 5.0,
) -> Vector:
    """
    Projection onto circle.

    Parameters:
        y (Vector): Point to project (NumPy array of length 2).
        center (Vector, optional): Centre of circle. Defaults to np.array([0.0, 0.0]).
        radius (Scalar, optional): Radius of circle. Defaults to 5.0.

    Returns:
        y_proj (Vector): Projection of `y` on the closed Euclidean ball (NumPy array of length 2).
    """

    direction: Vector = y - center
    distance: Scalar = Scalar(np.linalg.norm(direction))

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
        y (Vector): Point to project (NumPy array of length 2).
        low (Vector, optional): Lower corner of box. Defaults to np.array([-3.0, 0.0]).
        high (Vector, optional): Upper corner of box. Defaults to np.array([3.0, 4.0]).

    Returns:
        y_proj (Vector): Projection of `y` on the box (NumPy array of length 2).
    """

    y_proj: Vector = np.minimum(np.maximum(y, low), high)
    return y_proj


def SEPARATE_HYPERPLANE(
    constraints_CA: Optional[
        Callable[[cp.Variable], cp.Constraint | List[cp.Constraint]]
    ] = None,
    constraints_CB: Optional[
        Callable[[cp.Variable], cp.Constraint | List[cp.Constraint]]
    ] = None,
) -> Tuple[Vector, Scalar, Tuple[Vector, Vector]]:
    """
    Separating hyperplane (geometry / classification).

    Finds a hyperplane that separates two convex sets `C_A` and `C_B`.\\
    Canonical instance: unit circle vs half-space.

    Parameters
    ----------
        constraints_CA (Callable[[cp.Variable], cp.Constraint | List[cp.Constraint]], optional):
            A function that takes a CVXPY variable and returns a CVXPY constraint or a list of CVXPY constraints, defining set `C_A`.
            If None, uses the canonical instance (unit circle). Defaults to None.
        constraints_CB (Callable[[cp.Variable], cp.Constraint | List[cp.Constraint]], optional):
            A function that takes a CVXPY variable and returns a CVXPY constraint or a list of CVXPY constraints, defining set `C_B`.
            If None, uses the canonical instance (half-space). Defaults to None.

    Returns
    -------
        n (Vector): Normal vector of hyperplane (NumPy array of length 2).
        c (Scalar): Offset (scalar) so that hyperplane is {x: n^T x = c}.
        a_closest, b_closest (tuple[Vector, Vector]): The closest points in `C_A` and `C_B` used to construct the hyperplane.
    """

    a = cp.Variable(2)
    b = cp.Variable(2)

    if constraints_CA is not None and constraints_CB is not None:

        def cons_CA(x: cp.Variable) -> cp.Constraint | List[cp.Constraint]:
            return constraints_CA(x)

        def cons_CB(x: cp.Variable) -> cp.Constraint | List[cp.Constraint]:
            return constraints_CB(x)

    else:
        ## Canonical instance

        # Unit circle, ||x||_2 <= 1
        def cons_CA(x: cp.Variable) -> cp.Constraint | List[cp.Constraint]:
            return cp.norm2(x) <= 1

        # Half-space, x1 >= 3
        def cons_CB(x: cp.Variable) -> cp.Constraint | List[cp.Constraint]:
            return x[0] >= 3.0

    try:
        consA = cons_CA(a)
        consB = cons_CB(b)
    except Exception as e:
        raise ValueError("Possible invalid callables") from e

    constraints: List[cp.Constraint] = []
    for cons in [consA, consB]:
        if isinstance(cons, list):
            constraints.extend(cons)
        else:
            constraints.append(cons)

    objective = cp.Minimize(cp.norm(a - b))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimisation failed with status: {problem.status}")

    a_closest: Vector = np.asarray(a.value, dtype=np.double).flatten()
    b_closest: Vector = np.asarray(b.value, dtype=np.double).flatten()

    # Normal vector
    n: Vector = a_closest - b_closest
    n: Vector = n / np.linalg.norm(n)  # Normalise

    # Offset
    m: Vector = (a_closest + b_closest) / 2  # Midpoint
    c: Scalar = Scalar(n @ m)

    return n, c, (a_closest, b_closest)


def CHECK_FARKAS(
    A: Optional[Matrix] = None,
    b: Optional[Vector] = None,
) -> Tuple[bool, Optional[Vector], Dict[str, Any]]:
    """
    Farkas lemma / infeasibility check.

    Considers the feasibility problem `find x such that A x <= b`.\\
    If infeasible, finds a Farkas certificate `y` satisfying `y >= 0`, `A^T y = 0` (numerically), and `b^T y < 0` (numerically).

    Parameters
    ----------
        A (Matrix, optional): Coefficient matrix. Shape (m, n). If None, uses default instance. Defaults to None.
        b (Vector, optional): Right-hand side vector. Shape (m,). If None, uses default instance. Defaults to None.

    Returns
    -------
        feasible (bool): Boolean flag (True if feasible).
        y_cert (Vector, optional): If infeasible, return a Farkas certificate `y_cert`, else None.
        info (dict): Diagnostic info (objective value, solver status).
    """

    feasible: bool = False
    y_cert: Optional[Vector] = None
    info: Dict[str, Any] = {}

    if A is None:
        A = np.array([[1, 1], [-1, 0], [0, -1]], dtype=np.double)
    if b is None:
        b = np.array([-1, 0, 0], dtype=np.double)

    m, n = A.shape
    assert b.shape == (m,), "Incompatible dimensions between A and b."

    # Primal feasibility problem
    x = cp.Variable(n)
    primal_constraints: List[cp.Constraint] = [A[i] @ x <= b[i] for i in range(m)]
    primal_objective = cp.Minimize(0)
    primal_problem = cp.Problem(primal_objective, primal_constraints)
    primal_problem.solve()

    info.update({"primal_status": primal_problem.status})
    info.update({"primal_objective_value": primal_problem.value})
    info.update({"primal_solver_stats": primal_problem.solver_stats})

    if primal_problem.status in ["infeasible", "infeasible_inaccurate"]:
        # Farkas certificate problem
        y = cp.Variable(m, nonneg=True)
        dual_objective = cp.Maximize(b.T @ y)
        dual_constraints: List[cp.Constraint] = [A.T @ y == 0]
        dual_problem = cp.Problem(dual_objective, dual_constraints)
        dual_problem.solve()

        if dual_problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimisation failed with status: {dual_problem.status}")

        y_cert = np.array(y.value, dtype=np.double).flatten()
        ATy: Vector = A.T @ y_cert

        info.update({"dual_status": dual_problem.status})
        info.update({"dual_objective_value": dual_problem.value})  # b^T y_cert
        info.update({"dual_solver_stats": dual_problem.solver_stats})
        info.update({"dual_constraints_residual": ATy})
        info.update({"dual_constraints_residual_norm": np.linalg.norm(ATy)})

    else:
        feasible = True

    return feasible, y_cert, info


# ---------- Questions ----------
def question_1(X: Matrix, y: Vector, lambdas: List[Scalar]):
    print("\n\033[1m\033[4mQuestion-1\033[0m:")

    ## Q1 Part 3
    print("\033[4mPart-3\033[0m:")

    nonzero_counts: List[np.bool] = []
    for lam in lambdas:
        beta_star = LASSO_REGRESSION(X, y, lam)
        print(f"\u03bb = {lam}")
        print("Estimated coefficients \u03b2\u2217 =")
        print(beta_star, end="\n\n")

        nonzero_count = np.sum(np.abs(beta_star) > lam)
        nonzero_counts.append(nonzero_count)

    # Plot for sparsity of beta
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzero_counts, marker="o", linestyle="--")
    plt.xscale("log")
    plt.ylim(0, 17)
    plt.xlabel(r"$\lambda$ (Regularisation parameter)")
    plt.ylabel(r"Number of nonzero coefficients in $\beta^\ast$")
    plt.title(r"Sparsity of $\beta^\ast$ in LASSO Regression")
    plt.grid(True)
    if SAVE_FIGS:
        filename = f"Q1-3-Sparsity-LASSO-{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))

    ## Q1 Part 5
    # Duplicate one feature column in X and repeat the experiment in Part 3
    print("\033[4mPart-5\033[0m:")
    print("Duplicated feature column index:", DUP_COL_IDX, end="\n\n")
    X_dup = np.hstack((X, X[:, DUP_COL_IDX : DUP_COL_IDX + 1]))

    nonzero_counts_dup: List[np.bool] = []
    for lam in lambdas:
        beta_star = LASSO_REGRESSION(X_dup, y, lam)
        print(f"\u03bb = {lam}")
        print("Estimated coefficients \u03b2\u2217 =")
        print(beta_star, end="\n\n")

        nonzero_count = np.sum(np.abs(beta_star) > lam)
        nonzero_counts_dup.append(nonzero_count)

    # Plot for sparsity of beta
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzero_counts_dup, marker="o", linestyle="--")
    plt.xscale("log")
    plt.ylim(0, 17)
    plt.xlabel(r"$\lambda$ (Regularisation parameter)")
    plt.ylabel(r"Number of nonzero coefficients in $\beta^\ast$")
    plt.title(
        rf"Sparsity of $\beta^\ast$ in LASSO Regression with a duplicated feature (column {DUP_COL_IDX})"
    )
    plt.grid(True)
    if SAVE_FIGS:
        filename = f"Q1-5-Sparsity-LASSO-DupFeature-{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))


def question_2(X: Matrix, y: Vector, lambdas: List[Scalar]):
    print("\033[1m\033[4mQuestion-2\033[0m:")

    ## Q2 Part 3
    print("\033[4mPart-3\033[0m:")
    for lam in lambdas:
        u_star = LASSO_REGRESSION_DUAL(X, y, lam)
        print(f"\u03bb = {lam}")
        print("Estimated coefficients u\u2217 =")
        print(u_star, end="\n\n")

    ## Q2 Part 4
    # Closeness of optimum values u_star and beta_star
    print("\033[4mPart-4\033[0m:")
    for lam in lambdas:
        beta_star = LASSO_REGRESSION(X, y, lam)
        u_star = LASSO_REGRESSION_DUAL(X, y, lam)
        relation_lhs = X.T @ u_star
        relation_rhs = lam * np.sign(beta_star)
        l2_norm_diff = np.linalg.norm(relation_lhs - relation_rhs)
        print(f"\u03bb = {lam}")
        print(
            f"\u2016X\u1d40u\u2217 \u2013 \u03bb sign(\u03b2\u2217)\u2016\u2082 = {l2_norm_diff}",
            end="\n\n",
        )


def question_3():
    print("\033[1m\033[4mQuestion-3\033[0m:")

    ## Q3 Part 1
    # Projections in a navigation problem
    print("\033[4mPart-1\033[0m:")
    points: List[Vector] = [
        np.array([-2, -6]),
        np.array([-2, 6]),
        np.array([-4, -1]),
        np.array([-5, 3]),
        np.array([1, -2]),
        np.array([2, 3]),
        np.array([4, 1]),
        np.array([6, 6]),
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Projections onto Circle ($C_1$) and Box ($C_2$)")

    # Safe zones C_1, C_2
    ax.add_artist(Circle((0, 0), 5, color="lightblue", alpha=0.5))
    ax.add_patch(Rectangle((-3, 0), 6, 4, color="orange", alpha=0.5))

    for p in points:
        proj_circle = PROJ_CIRCLE(p)
        proj_box = PROJ_BOX(p)

        ax.plot(*p, "g*", zorder=3, markersize=16)

        # Projections
        ax.plot(*proj_circle, "bo", zorder=5, markersize=6)
        ax.plot(*proj_box, "rs", zorder=4, markersize=8)

        # Arrows
        ax.arrow(*p, *(proj_box - p), color="red", linestyle="--", alpha=0.6, zorder=2)
        ax.arrow(
            *p, *(proj_circle - p), color="blue", linestyle="--", alpha=0.6, zorder=2
        )
    ax.legend(
        [
            r"Circle ($C_1$)",
            r"Box ($C_2$)",
            "Original point",
            r"Projection on $C_1$",
            r"Projection on $C_2$",
        ]
    )
    plt.grid(True)
    plt.tight_layout()
    print("Plot generated for projections onto circle and box.")
    if SAVE_FIGS:
        filename = f"Q3-1-Projections-{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))

    ## Q3 Part 2
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
        # vertical line x1 = c / n1
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
    if SAVE_FIGS:
        filename = f"Q3-2-Separating-Hyperplane-{SRN}.jpeg"
        plt.savefig(os.path.join(FIGS_DIR, filename))

    ## Q3 Part 3
    # Farkas lemma in a supply-chain model
    print("\n\033[4mPart-3\033[0m:")
    feasible, y_cert, info = CHECK_FARKAS()
    print("Is feasible:", feasible)
    print("Certificate y:", y_cert)
    print("Diagnostics:", info)


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")
    print("Available CVXPY solvers:", cp.installed_solvers())

    f1(SRN)
    filename = f"data_{SRN}.csv"
    data = pd.read_csv(filename)
    X: Matrix = data.values[:, :15]
    y: Vector = data.values[:, 15]

    lambdas: List[Scalar] = [0.01, 0.1, 1]

    question_1(X, y, lambdas)
    question_2(X, y, lambdas)
    question_3()

    if not SAVE_FIGS:
        plt.show()
    plt.close("all")
