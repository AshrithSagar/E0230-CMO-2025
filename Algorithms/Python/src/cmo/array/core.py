"""
Core array interfaces
=======
src/cmo/array/core.py
"""

from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

_Shape: TypeAlias = tuple[Any, ...]
_AnyShape: TypeAlias = tuple[Any, ...]

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=Any, default=Any, covariant=True)


@runtime_checkable
class TypedArray(Protocol[_ShapeT_co, _DTypeT_co]):
    @property
    def shape(self) -> _ShapeT_co: ...

    @property
    def dtype(self) -> _DTypeT_co: ...
