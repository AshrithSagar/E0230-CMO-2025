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

Vector: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 1D numpy array of real numbers."""

Matrix: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 2D numpy array of real numbers"""
