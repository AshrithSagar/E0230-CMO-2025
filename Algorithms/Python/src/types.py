"""
src/types.py
=======
Type aliases and utility types
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

floatVec: TypeAlias = npt.NDArray[np.float64]
"""A type alias for a 1D numpy array of real numbers."""

floatMat: TypeAlias = npt.NDArray[np.float64]
"""A type alias for a 2D numpy array of real numbers"""
