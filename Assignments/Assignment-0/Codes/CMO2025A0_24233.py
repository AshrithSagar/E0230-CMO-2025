import os
import sys

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

a, b = oracle(12345, 42)
print(f"{a=}, {b=}")
