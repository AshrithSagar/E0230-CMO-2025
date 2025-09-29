Oracle package for CMO Assignment 2025A2.

Tested with: Python 3.10.12 (Conda env `oracle310`)
Install requirements:
    python -m pip install -r requirements.txt

Usage (after unzipping):
    from oracle_final import f2, f5
    A, b = f5(12345)
    A, b = f2(12345, True)
Notes:
    - A returned by f5 is a LinearOperator; use A @ x or A.dot(x).
