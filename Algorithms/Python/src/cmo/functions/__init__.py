"""
Optimisation functions
=======
src/cmo/functions/__init__.py

References
-------
- https://www.sfu.ca/~ssurjano/optimization.html
- https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from cmo.functions.base import ConvexQuadratic, Function, LinearFunction
from cmo.functions.test import (
    Ackley,
    Beale,
    Booth,
    DropWave,
    Eggholder,
    Griewank,
    Levy,
    Levy13,
    Matyas,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    SumSquares,
    ThreeHumpCamel,
    Zakharov,
)

__all__ = [
    "Function",
    "LinearFunction",
    "ConvexQuadratic",
    "Ackley",
    "Beale",
    "Booth",
    "DropWave",
    "Eggholder",
    "Griewank",
    "Levy",
    "Levy13",
    "Matyas",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "SumSquares",
    "ThreeHumpCamel",
    "Zakharov",
]
