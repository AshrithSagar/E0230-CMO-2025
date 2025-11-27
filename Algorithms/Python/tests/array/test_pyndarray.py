"""
tests/array/test_pyndarray.py
"""

from cmo.array.python import PyNDArray


def test1():
    a = PyNDArray([1, 2, 3])
    assert a.dtype == int
    assert a.shape == (3,)


def test2():
    a = PyNDArray([[1.0, 2.0], [3.0, 4.0]])
    assert a.dtype == float
    assert a.shape == (2, 2)


if __name__ == "__main__":
    test1()
    test2()
