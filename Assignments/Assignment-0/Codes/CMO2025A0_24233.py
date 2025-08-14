# ---------- CMO 2025 Assignment 0 ----------

# ---------- Imports ----------
import os
import sys
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

# ---------- Setup ----------
SRN = 24233
"""
The 5-digit Student Registration Number (SRN) for the assignment.
"""
assert isinstance(SRN, int) and len(str(SRN)) == 5, "SRN must be a 5-digit integer."


# ---------- Oracle utils ----------
def oracle_f(x: float) -> tuple[float, float]:
    """
    A wrapper around the provided `oracle` function.

    `f(x), f'(x) = oracle_f(x)`
    """
    return oracle(SRN, x)


FirstOrderOracleFn: TypeAlias = Callable[[float], tuple[float, float]]
"""
Type alias for a first-order oracle function.
A function of this type takes a float `x` and returns a tuple `(f(x), f'(x))`.
"""


# ---------- Iterative Algorithm Template ----------
class IterativeOptimiser(ABC):
    """
    A base template class for iterative optimisation algorithms,
    particularly for the minimisation objective.

    `x^{k+1} = ALGO(x^k, f'(x^k), k)`
    where `ALGO` is the algorithm-specific step function,
    `x^k` is the value at iteration `k`,
    """

    def __init__(self, name: str, config: dict):
        """
        Parameters:
            name: Name of the algorithm.
            config: Configuration parameters for the algorithm.
        """
        self.name = name
        self.config = config

        self.history: list[float] = []
        self.x_star: float
        self.fx_star: float
        self.dfx_star: float

    def run(
        self, oracle_fn: FirstOrderOracleFn, x0: float, maxiter=1_000_000, tol=1e-6
    ):
        """
        Runs the iterative algorithm.

        Parameters:
            oracle_fn: The first-order oracle function to minimize.
            x0: Initial guess for the minimum point.
            maxiter: Maximum number of iterations to perform.
            tol: Tolerance for stopping criterion based on the gradient.
        """
        self.oracle_fn = oracle_fn
        self.history = [x0]
        x = x0
        self._initialize_state()

        for t in range(1, maxiter + 1):
            fx, dfx = self.oracle_fn(x)  # Query the oracle function
            if abs(dfx) < tol:  # Early exit if f'(x) is small enough
                break
            x = self._step(x, dfx, t)
            self.history.append(x)

        self.x_star = x
        self.fx_star, self.dfx_star = self.oracle_fn(x)

    @abstractmethod
    def _initialize_state(self):
        """
        Initializes the state of the algorithm.
        This method can be overridden by subclasses to set up any necessary state.
        """
        pass

    @abstractmethod
    def _step(self, x: float, grad: float, t: int) -> float:
        """
        Performs a single step of the algorithm.
        Parameters:
            x: Current value of `x`.
            grad: Current gradient `f'(x)`.
            t: Current iteration number.
        Returns:
            The updated value of `x` after the step.
        """
        pass

    def summary(self):
        """Prints a summary of the algorithm's results."""
        print(f"\n{self.name}")
        print(
            f"x* = {self.x_star:.6f}, f(x*) = {self.fx_star:.6f}, f'(x*) = {self.dfx_star:.2e}"
        )
        if abs(self.dfx_star) < 1e-6:
            print("Success! Gradient is close to zero.")
        else:
            print("Error: Gradient is not close to zero.")

    def plot(self):
        """Plots the history of `x` values during the optimisation."""
        plt.plot(self.history, label=self.name)


# ---------- Optimiser Implementations ----------
class GradientDescent(IterativeOptimiser):
    def _initialize_state(self):
        pass

    def _step(self, x, grad, t):
        return x - self.config["lr"] * grad


class MomentumGradientDescent(IterativeOptimiser):
    def _initialize_state(self):
        self.v = 0.0

    def _step(self, x, grad, t):
        self.v = self.config["momentum"] * self.v - self.config["lr"] * grad
        return x + self.v


class Adagrad(IterativeOptimiser):
    def _initialize_state(self):
        self.gsum = 0.0

    def _step(self, x, grad, t):
        self.gsum += grad**2
        adjusted_lr = self.config["lr"] / (self.gsum**0.5 + self.config["eps"])
        return x - adjusted_lr * grad


class RMSProp(IterativeOptimiser):
    def _initialize_state(self):
        self.avg_sq_grad = 0.0

    def _step(self, x, grad, t):
        beta = self.config["beta"]
        self.avg_sq_grad = beta * self.avg_sq_grad + (1 - beta) * grad**2
        return x - self.config["lr"] * grad / (
            self.avg_sq_grad**0.5 + self.config["eps"]
        )


class Adam(IterativeOptimiser):
    def _initialize_state(self):
        self.m = 0.0
        self.v = 0.0

    def _step(self, x, grad, t):
        b1, b2 = self.config["beta1"], self.config["beta2"]
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad**2

        m_hat = self.m / (1 - b1**t)
        v_hat = self.v / (1 - b2**t)

        return x - self.config["lr"] * m_hat / (v_hat**0.5 + self.config["eps"])


# ---------- Main ----------
if __name__ == "__main__":
    optimizers: list[IterativeOptimiser] = [
        GradientDescent("Gradient Descent", {"lr": 1e-3}),
        MomentumGradientDescent("Momentum-based GD", {"lr": 1e-3, "momentum": 0.9}),
        Adagrad("Adagrad", {"lr": 1e-2, "eps": 1e-8}),
        RMSProp("RMSProp", {"lr": 1e-3, "beta": 0.9, "eps": 1e-8}),
        Adam("Adam", {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
    ]
    for opt in optimizers:
        opt.run(oracle_f, x0=0.0)
        opt.summary()

    # Plot
    plt.figure(figsize=(10, 6))
    for opt in optimizers:
        opt.plot()
    plt.xlabel("Iteration")
    plt.ylabel("x value")
    plt.title("Convergence of Optimisation Algorithms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
