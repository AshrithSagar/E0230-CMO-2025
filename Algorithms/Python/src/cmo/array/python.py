"""
PyNDArray's
=======
src/cmo/array/python.py
"""

from typing import Any, Iterable, Sequence

type RecursiveSequence[T] = Sequence[T | "RecursiveSequence[T]"]
type RecursiveList[T] = list[T | "RecursiveList[T]"]

# Numeric promotion hierarchy
_numeric_hierarchy = [int, float, complex]
_type_priority = {typ: i for i, typ in enumerate(_numeric_hierarchy)}


class PyNDArray[T = float]:
    def __init__(self, array: RecursiveSequence[T]) -> None:
        if not isinstance(array, list):
            raise TypeError("Input must be a list")
        self.array: RecursiveList[T] = self._to_list(array)
        self.shape = self._compute_shape(self.array)
        self.dtype = self._compute_dtype(self.array)

    def _to_list(self, x: Any) -> Any:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return [self._to_list(e) for e in x]
        return x

    def _compute_shape(self, x: Any) -> tuple[int, ...]:
        shape: list[int] = []
        while isinstance(x, list):
            shape.append(len(x))
            if len(x) == 0:
                break
            # Check that all sublists are the same length
            first_len = len(x[0]) if isinstance(x[0], list) else None
            for item in x:
                if isinstance(item, list) and len(item) != first_len:
                    raise ValueError("All sublists must have the same length")
                if not isinstance(item, list) and first_len is not None:
                    raise ValueError("Inconsistent nesting")
            x = x[0] if x else []
        return tuple(shape)

    def _compute_dtype(self, x: Any) -> type:
        if isinstance(x, list):
            if not x:
                return float
            dtypes: set[type] = {self._compute_dtype(e) for e in x}
            numeric_types: set[type] = {t for t in dtypes if t in _type_priority}
            if numeric_types:
                return max(numeric_types, key=lambda t: _type_priority[t])
            if len(set(dtypes)) == 1:
                return dtypes.pop()
            raise TypeError("All elements must have the same type")
            return object  # Fallback
        else:
            return type(x)

    def __str__(self) -> str:
        return f"array({self.array})"

    def __repr__(self) -> str:
        return f"PyNDArray(shape={self.shape}, dtype={self.dtype.__name__}, array={self.array})"
