# ---------- CMO 2025 Assignment 3 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.abspath("oracle_2025A3"))
from oracle_2025A3 import f1  # type: ignore

# Type aliases
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

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


# ---------- Implementations ----------
def LASSO_REGRESSION(X: Matrix, y: Vector, lam: float) -> Vector:
    """
    Solve the Lasso regression problem for the Linear objective function using CVXPY.

    `min_{beta} 0.5 ||X beta - y||_2^2 + lam ||beta||_1`

    Parameters:
        X (NDArray): Feature matrix. Shape (n_samples, n_features).
        y (NDArray): Response vector. Shape (n_samples,).
        lam (float): Regularisation parameter. Must be non-negative.

    Returns:
        beta (NDArray): Estimated coefficients.
    """

    assert lam >= 0, "Regularisation parameter must be non-negative."

    n_features = X.shape[1]
    beta = cp.Variable(n_features)

    objective = cp.Minimize(0.5 * cp.sum_squares(X @ beta - y) + lam * cp.norm1(beta))
    problem = cp.Problem(objective)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimisation failed with status: {problem.status}")

    return np.array(beta.value, dtype=np.float64).flatten()


def LASSO_REGRESSION_DUAL(X: Matrix, y: Vector, lam: float) -> Vector:
    """
    Solve the dual of the Lasso regression problem using CVXPY.

    `max_{u} -0.5 ||u||_2^2 + y'u`\\
    `subject to ||X'u||_infty <= lam`

    Parameters:
        X (NDArray): Feature matrix. Shape (n_samples, n_features).
        y (NDArray): Response vector. Shape (n_samples,).
        lam (float): Regularisation parameter. Must be non-negative.

    Returns:
        u (NDArray): Dual variable.
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

    return np.array(u.value, dtype=np.float64).flatten()


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
        # Normalise certificate
        if np.linalg.norm(A.T @ y) < 1e-6 and b @ y < -1e-6:
            return False, y, {"status": prob.status, "bTy": b @ y, "ATy": A.T @ y}
        else:
            return False, None, {"status": prob.status, "dual_failed": True}
    else:
        return True, None, {"status": prob.status}


# ---------- Questions ----------
def question_1(X: Matrix, y: Vector, lambdas: List[float]):
    print("\n\033[1m\033[4mQuestion-1\033[0m:")

    ## Q1 Part 3
    print("\033[4mPart-3\033[0m:")

    nonzero_counts: List[np.bool] = []
    for lam in lambdas:
        beta_star = LASSO_REGRESSION(X, y, lam)
        print(f"lambda: {lam}")
        print("Estimated coefficients (beta_star):")
        print(beta_star, end="\n\n")

        nonzero_count = np.sum(np.abs(beta_star) > lam)
        nonzero_counts.append(nonzero_count)

    # Plot for sparsity of beta
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzero_counts, marker="o", linestyle="--")
    plt.xscale("log")
    plt.ylim(0, 17)
    plt.xlabel(r"$\lambda$ (Regularisation parameter)")
    plt.ylabel(r"Number of nonzero coefficients in $\beta^*$")
    plt.title(r"Sparsity of $\beta^*$ in LASSO Regression")
    plt.grid(True)

    ## Q1 Part 5
    # Duplicate one feature column in X and repeat the experiment in Part 3
    print("\033[4mPart-5\033[0m:")
    print("Duplicated feature column index:", DUP_COL_IDX, end="\n\n")
    X_dup = np.hstack((X, X[:, DUP_COL_IDX : DUP_COL_IDX + 1]))

    nonzero_counts_dup: List[np.bool] = []
    for lam in lambdas:
        beta_star = LASSO_REGRESSION(X_dup, y, lam)
        print(f"lambda: {lam}")
        print("Estimated coefficients (beta_star):")
        print(beta_star, end="\n\n")

        nonzero_count = np.sum(np.abs(beta_star) > lam)
        nonzero_counts_dup.append(nonzero_count)

    # Plot for sparsity of beta
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzero_counts_dup, marker="o", linestyle="--")
    plt.xscale("log")
    plt.ylim(0, 17)
    plt.xlabel(r"$\lambda$ (Regularisation parameter)")
    plt.ylabel(r"Number of nonzero coefficients in $\beta^*$")
    plt.title(r"Sparsity of $\beta^*$ in LASSO Regression with a duplicated feature")
    plt.grid(True)


def question_2(X: Matrix, y: Vector, lambdas: List[float]):
    print("\033[1m\033[4mQuestion-2\033[0m:")

    ## Q2 Part 3
    print("\033[4mPart-3\033[0m:")
    for lam in lambdas:
        u_star = LASSO_REGRESSION_DUAL(X, y, lam)
        print(f"lambda: {lam}")
        print("Estimated coefficients (u_star):")
        print(u_star, end="\n\n")


def question_3():
    print("\033[1m\033[4mQuestion-3\033[0m:")

    ## Q3 Part 1
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

    lambdas: List[float] = [0.01, 0.1, 1]

    question_1(X, y, lambdas)
    question_2(X, y, lambdas)
    question_3()

    plt.show()
