# ---------- CMO 2025 Assignment 0 ----------

# ---------- Imports ----------
# Allowed libraries: os, sys, numpy, math, matplotlib.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

# ---------- Setup ----------
SRN = 24233
"""
The 5-digit Student Registration Number (SRN) for the assignment.
"""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."


# ---------- Oracle utils ----------
class FirstOrderOracle:
    """
    A wrapper class around the provided `oracle` function.

    `f(x), f'(x) = oracle(SRN, x)`
    """

    def __init__(self):
        self.call_count: int = 0
        """
        Tracks the number of times the oracle function has been called.
        This is useful for the 'analytical complexity' of the algorithms.
        """

    def __call__(self, x: float) -> tuple[float, float]:
        """Evaluates the oracle function at `x`."""
        self.call_count += 1
        return oracle(SRN, x)

    def reset_counter(self) -> "FirstOrderOracle":
        """Resets the internal call count to zero."""
        self.call_count = 0
        return self

    def plot(self, x_range: tuple[float, float], num_points=100):
        """
        Plots the oracle function over a specified range.\\
        Just for convenience, to visualise the function `f(x)`, quering the oracle multiple times.

        Parameters:
            x_range: A tuple specifying the range of `x` values to plot.
            num_points: Number of points to sample in the range.
        """
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = [self.__call__(x)[0] for x in x_values]

        plt.plot(x_values, y_values, label="f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Oracle Function Plot")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


# ---------- Iterative Algorithm Template ----------
class IterativeOptimiser:
    """
    A base template class for iterative optimisation algorithms,
    particularly for the minimisation objective.

    `x^{k+1} = ALGO(x^k, f'(x^k), k)`
    where `ALGO` is the algorithm-specific step function,
    `x^k` is the value at iteration `k`,
    """

    def __init__(self, **kwargs):
        # Initialises the iterative optimiser with configuration parameters.
        self.config = kwargs

        self.name = self.__class__.__name__
        """Name of the algorithm, derived from the class name of the optimiser."""

        self.history: list[float] = []
        self.x_star: float
        self.fx_star: float
        self.dfx_star: float

        self.oracle_fn: FirstOrderOracle
        self.x0: float
        self.maxiter: int
        self.tol: float

    def run(self, oracle_fn: FirstOrderOracle, x0: float, maxiter=1_000_000, tol=1e-9):
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimise.
            x0: Initial guess for the minimum point.
            maxiter: Maximum number of iterations to perform.
            tol: Tolerance for stopping criterion based on the gradient.
        """
        self.oracle_fn = oracle_fn.reset_counter()
        self.x0 = x0
        self.maxiter = maxiter
        self.tol = tol

        self.history = [x0]
        x = x0
        self._initialise_state()

        for t in range(1, maxiter + 1):
            fx, dfx = self.oracle_fn(x)  # Query the oracle function
            if abs(dfx) < tol:  # Early exit if f'(x) is small enough
                break
            x = self._step(x, dfx, t)
            self.history.append(x)

        self.x_star = x
        self.fx_star, self.dfx_star = self.oracle_fn(x)

    def _initialise_state(self) -> None:
        """
        Initialises the state of the algorithm.\\
        [Optional]: This method can be overridden by subclasses to set up any necessary state if needed.
        """
        pass

    def _step(self, x: float, grad: float, t: int) -> float:
        """
        Performs a single step of the algorithm.\\
        [Required]: This method should be implemented by subclasses to define the specific update rule.
        Parameters:
            x: Current value of `x`.
            grad: Current gradient `f'(x)`.
            t: Current iteration number.
        Returns:
            The updated value of `x` after the step.
        """
        raise NotImplementedError

    def summary(self):
        """Prints a summary of the algorithm's results."""
        num_iters = len(self.history) - 1
        converged = abs(self.dfx_star) < self.tol
        oracle_calls = self.oracle_fn.call_count

        print(f"\n{self.name}")
        print(f"x* = {self.x_star}, f(x*) = {self.fx_star}, f'(x*) = {self.dfx_star}")
        print(
            f"Iterations: {num_iters} {'(Converged)' if converged else '(Did NOT converge)'}"
        )
        print(f"Number of calls to the oracle: {oracle_calls}")

        if converged:
            print(f"Gradient is close to zero within the tolerance ({self.tol}).")
        else:
            print(f"Did not converge within {self.maxiter} iterations.")

    def plot(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)


# ---------- Optimiser Implementations ----------
class GradientDescent(IterativeOptimiser):
    """
    Standard Gradient Descent.

    `x_{k+1} = x_k - eta f'(x_k)`\\
    where `eta` is the learning rate.
    """

    def _step(self, x, grad, t):
        eta: float = self.config["lr"]
        return x - eta * grad


class MomentumGradientDescent(IterativeOptimiser):
    """
    Gradient Descent with Momentum.

    `v_{k+1} = gamma v_k - eta f'(x_k)`\\
    `x_{k+1} = x_k + v_{k+1}`\\
    where `gamma` is the momentum parameter and `eta` is the learning rate.
    """

    def _initialise_state(self):
        self.v = 0.0

    def _step(self, x, grad, t):
        gamma: float = self.config["momentum"]
        eta: float = self.config["lr"]

        self.v = gamma * self.v - eta * grad
        return x + self.v


class BacktrackingGradientDescent(IterativeOptimiser):
    """
    Gradient Descent with Backtracking Line Search.

    Starts with an initial learning rate and reduces it using Armijo condition:
    `f(x - eta * grad) <= f(x) - alpha * eta * grad^2`
    where `eta` is the step size, `alpha` is a constant, and `beta` is the reduction factor.
    The step size is reduced until the condition is satisfied.
    """

    def _step(self, x, grad, t):
        eta = self.config["init_lr"]
        alpha = self.config["alpha"]
        beta = self.config["beta"]

        f_x, _ = self.oracle_fn(x)
        while True:
            x_new = x - eta * grad
            f_new, _ = self.oracle_fn(x_new)  # Needs multiple queries

            # Armijo condition
            if f_new <= f_x - alpha * eta * grad**2:
                break  # Step size is good enough
            eta *= beta  # Reduce step size

        return x - eta * grad  # Step


class BFGS(IterativeOptimiser):
    """
    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm
    which approximates the inverse Hessian.\\
    This is a quasi-Newton algorithm, and the 1D case is implemented here.

    `x_{k+1} = x_k - H_k * f'(x_k)`\\
    where `H_k` is the inverse Hessian approximation at iteration `k`.

    `H_init` is the initial inverse Hessian approximation.
    """

    def _initialise_state(self):
        self.H = self.config.get("H_init", 1.0)
        self.prev_grad = None
        self.prev_x = None

    def _step(self, x, grad, t):
        if self.prev_x is not None and self.prev_grad is not None:
            s = x - self.prev_x
            y = grad - self.prev_grad
            denom = s * y

            if denom != 0:  # Scalar update in 1D
                self.H = (s**2) / denom

        # Update current values
        self.prev_x = x
        self.prev_grad = grad

        return x - self.H * grad  # Step


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    oracle_f = FirstOrderOracle()
    oracle_f.plot(x_range=(-20, 20))

    optimisers: list[IterativeOptimiser] = [
        GradientDescent(lr=1e-3),
        MomentumGradientDescent(lr=1e-3, momentum=0.9),
        BacktrackingGradientDescent(init_lr=0.5, alpha=0.1, beta=0.9),
        BFGS(H_init=1.0),
    ]
    for opt in optimisers:
        opt.run(oracle_f, x0=0.0, maxiter=1_000_000, tol=1e-12)
        opt.summary()

    # Convergence of optimisers plot
    plt.figure()
    for opt in optimisers:
        opt.plot()
    plt.xlabel("Iteration")
    plt.ylabel("x value")
    plt.title("Convergence of Optimisation Algorithms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
