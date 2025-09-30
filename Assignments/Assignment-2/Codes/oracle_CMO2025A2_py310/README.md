# Oracle 2025A2

This package provides the obfuscated oracle functions *f2* and *f5* for Assignment 2.  
The same archive works on *Linux* and *macOS (Intel & Apple Silicon)*.  
Windows users must run inside *WSL (Windows Subsystem for Linux)*.

---

## Contents

- oracle_2025A2/
  - oracle_final.py.
  - pyarmor_runtime_000000/.
  - requirements.txt — dependencies required to run the oracle.
  - README.md — this instruction file.

---

## Installation & Usage

1. Ensure you are using *Python 3.10* (recommended: create a conda environment).
   ```bash
   conda create -n oracle310 python=3.10
   conda activate oracle310
   ```
   
#Install required packages
pip install -r requirements.txt

In your python scripts, import the oracle:

from oracle_final import f2, f5

# Example usage:
A, b = f2(srno=12345, subq=True)   # Q1 oracle (SPD matrix + b)
A, b = f5(srno=12345)              # Q2 oracle (LinearOperator + b)
