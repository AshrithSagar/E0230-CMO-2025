"""
Optimiser implementations
=======
src/cmo/optimisers/__init__.py

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

from .conjugate_gradient import ConjugateDirectionMethod, ConjugateGradientMethod
from .line_search import (
    GradientDescent,
    GradientDescentArmijo,
    GradientDescentArmijoGoldstein,
    GradientDescentBacktracking,
    GradientDescentExactLineSearch,
    GradientDescentWolfe,
)
from .quasi_newton import BFGSUpdate, DFPUpdate, NewtonMethod, SR1Update

__all__ = [
    "ConjugateDirectionMethod",
    "ConjugateGradientMethod",
    "GradientDescent",
    "GradientDescentArmijo",
    "GradientDescentArmijoGoldstein",
    "GradientDescentBacktracking",
    "GradientDescentExactLineSearch",
    "GradientDescentWolfe",
    "BFGSUpdate",
    "DFPUpdate",
    "NewtonMethod",
    "SR1Update",
]
