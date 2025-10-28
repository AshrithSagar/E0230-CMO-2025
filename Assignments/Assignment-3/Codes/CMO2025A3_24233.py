# ---------- CMO 2025 Assignment 3 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable

import numpy as np
import numpy.typing as npt

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


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    f1(SRN)
