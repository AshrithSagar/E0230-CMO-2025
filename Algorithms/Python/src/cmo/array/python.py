"""
PyNDArray's
=======
src/cmo/array/python.py

Multi-dimensional array implementation in pure Python.
This is just an experiment to implement numpy-like arrays in pure Python for educational purposes.
Do not use this in production; prefer numpy instead.
"""

from typing import Any, Generic, Iterable, SupportsFloat, TypeAlias, TypeVar, cast

_ShapeT_co = TypeVar(
    "_ShapeT_co", bound=tuple[int, ...], default=tuple[int, ...], covariant=True
)
_DTypeT_co = TypeVar("_DTypeT_co", bound=SupportsFloat, default=float, covariant=True)

# Numeric promotion hierarchy
_numeric_hierarchy = [int, float]
_type_priority = {typ: i for i, typ in enumerate(_numeric_hierarchy)}

_RecursiveList: TypeAlias = list["_DTypeT_co | _RecursiveList[_DTypeT_co]"]


class PyNDArray(Generic[_ShapeT_co, _DTypeT_co]):
    array: _RecursiveList[_DTypeT_co]

    def __init__(self, array: _RecursiveList[_DTypeT_co]) -> None:
        self.array = self._validate_array(array)

    @property
    def shape(self) -> _ShapeT_co:
        return cast(_ShapeT_co, self._compute_shape(self.array))

    @property
    def dtype(self) -> _DTypeT_co:
        return cast(_DTypeT_co, self._compute_dtype(self.array))

    def __getitem__(self, index: int | slice) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"array({self.array})"

    def __repr__(self) -> str:
        return f"PyNDArray(shape={self.shape}, dtype={self.dtype}, array={self.array})"

    def _validate_array(self, x: Any) -> Any:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return [self._validate_array(e) for e in x]
        return x

    def _compute_shape(self, x: Any) -> tuple[int, ...]:
        shape = list[int]()
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
            if len(dtypes) == 1:
                return dtypes.pop()
            raise TypeError("All elements must have the same type")
            return object  # Fallback
        else:
            return type(x)

    def astype(self, dtype: type) -> None:
        def _astype(x: Any, dtype: type) -> Any:
            if isinstance(x, list):
                return [_astype(e, dtype) for e in x]
            else:
                return dtype(x)

        self.array = _astype(self.array, dtype)
