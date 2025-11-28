"""
Type aliases and utility types
=======
src/cmo/types.py
"""

from typing import TypeAlias

import numpy as np

from cmo.array.numpy import TypedNDArray

## Helpers

dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""

# Shape type aliases
Shape1D: TypeAlias = tuple[int]
"""A tuple representing a 1D shape, i.e., `(N,)`."""
Shape2D: TypeAlias = tuple[int, int]
"""A tuple representing a 2D shape, i.e., `(M, N)`."""

# Array type aliases
Vector: TypeAlias = TypedNDArray[Shape1D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
Matrix: TypeAlias = TypedNDArray[Shape2D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""

Scalar: TypeAlias = float
"""A type alias for a scalar real number."""
