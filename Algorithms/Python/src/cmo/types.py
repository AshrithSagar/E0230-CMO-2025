"""
Type aliases and utility types
=======
src/cmo/types.py
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Scalar: TypeAlias = float
"""A type alias for a scalar real number."""

dtype: TypeAlias = np.double
"""A type alias for the numpy data type representing real numbers."""

Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[dtype]]
"""A type alias for a 1D numpy array of real numbers."""

Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[dtype]]
"""A type alias for a 2D numpy array of real numbers"""


def asVector(x: npt.NDArray, dtype=dtype) -> Vector:
    """Helper to convert a numpy ndarray to a `Vector` (1D numpy array)."""
    assert x.ndim == 1, "Input array must be 1-dimensional."
    return x.astype(dtype)


def asMatrix(x: npt.NDArray, dtype=dtype) -> Matrix:
    """Helper to convert a numpy ndarray to a `Matrix` (2D numpy array)."""
    assert x.ndim == 2, "Input array must be 2-dimensional."
    return x.astype(dtype)
