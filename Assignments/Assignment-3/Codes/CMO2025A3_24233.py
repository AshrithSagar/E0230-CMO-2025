# ---------- CMO 2025 Assignment 3 ----------

# ---------- Imports ----------
import os
import sys
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

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


# ---------- Questions ----------
def question_1():
    print("\n\033[1m\033[4mQuestion-1\033[0m:")

    f1(SRN)
    filename = f"data_{SRN}.csv"
    data = pd.read_csv(filename)
    X: Matrix = data.values[:, :15]
    y: Vector = data.values[:, 15]


def question_2():
    print("\n\033[1m\033[4mQuestion-2\033[0m:")


def question_3():
    print("\n\033[1m\033[4mQuestion-3\033[0m:")


# ---------- Main ----------
if __name__ == "__main__":
    print(f"{SRN = }")

    question_1()
    question_2()
    question_3()
