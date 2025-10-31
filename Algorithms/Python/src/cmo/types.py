"""
Type aliases and utility types
=======
src/cmo/types.py
"""

from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt

floatVec: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 1D numpy array of real numbers."""

floatMat: TypeAlias = npt.NDArray[np.double]
"""A type alias for a 2D numpy array of real numbers"""

floatOrVec = Union[float, floatVec]
"""A type alias for either a scalar or a vector of floats."""
