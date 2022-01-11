"""Internal types module"""
from __future__ import annotations

import sys
from abc import ABC, ABCMeta, abstractmethod
from collections import deque
from functools import reduce, wraps
from itertools import islice
from os import PathLike
from typing import (
    Any, Callable, Collection, Deque, Dict, Generic, Iterable, Iterator, Reversible, Sequence,
    SupportsIndex, Tuple, TypeVar, Union, cast, get_args, get_origin, overload
)

from numpy.typing import NDArray
from typing_extensions import Annotated, get_type_hints

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
TCV_co = TypeVar('TCV_co', bound=Union[float, int, str], covariant=True)  # Type Color Value covariant
TCV_inv = TypeVar('TCV_inv', bound=Union[float, int, str])  # Type Color Value invariant
ACV = Union[float, int, str]
Nb = TypeVar('Nb', bound=Union[float, int])  # Number
Tup3 = Tuple[Nb, Nb, Nb]
Tup4 = Tuple[Nb, Nb, Nb, Nb]
Tup3Str = Tuple[str, str, str]
if sys.version_info >= (3, 9):
    AnyPath = Union[PathLike[str], str]
else:
    AnyPath = Union[PathLike, str]
SomeArrayLike = Union[Sequence[float], NDArray[Any]]


class CheckAnnotated(Generic[T], ABC):
    @abstractmethod
    def check(self, val: T | Iterable[T], param_name: str) -> None:
        ...


class ValueRangeInclExcl(CheckAnnotated[float]):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def check(self, val: float | Iterable[float], param_name: str) -> None:
        val = [val] if isinstance(val, float) else val
        for v in val:
            if not self.x < v <= self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


class ValueRangeIncInc(CheckAnnotated[float]):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def check(self, val: float | Iterable[float], param_name: str) -> None:
        val = [val] if isinstance(val, float) else val
        for v in val:
            if not self.x < v < self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


Nb8bit = Annotated[int, ValueRangeInclExcl(0, 256)]
Nb16bit = Annotated[int, ValueRangeInclExcl(0, 65536)]
NbFloat = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Pct = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Alignment = Annotated[int, ValueRangeIncInc(0, 9)]


def check_annotations(func: F, /) -> F:

    def _check_hint(hint: Any, value: Any, param_name: str) -> None:
        if get_origin(hint) is Annotated:
            # hint_type, *hint_args = get_args(hint)
            _, *hint_args = get_args(hint)
            for hint_arg in hint_args:
                if isinstance(hint_arg, CheckAnnotated):
                    hint_arg.check(value, param_name)
                else:
                    raise TypeError

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        type_hints = get_type_hints(func, include_extras=True)
        for value, (param_name, hint) in zip(list(args) + list(kwargs.values()), type_hints.items()):
            for h in hint.__args__:
                _check_hint(h, value, param_name)
        return func(*args, **kwargs)

    return cast(F, wrapper)


class View(Reversible[T], Collection[T]):
    """Abstract View class"""
    __slots__ = '__x'

    def __init__(self, __x: Collection[T]) -> None:
        self.__x = __x
        super().__init__()

    def __contains__(self, __x: object) -> bool:
        return __x in self.__x

    def __iter__(self) -> Iterator[T]:
        return iter(self.__x)

    def __len__(self) -> int:
        return len(self.__x)

    def __reversed__(self) -> Iterator[T]:
        return reversed(tuple(self.__x))

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.__x})'

    __repr__ = __str__


class NamedMutableSequenceMeta(ABCMeta):
    __slots__: Tuple[str, ...] = ()

    def __new__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any],
                ignore_slots: bool = False, **kwargs: Any) -> NamedMutableSequenceMeta:
        # Let's use __slots__ only if the class is a concrete application
        if ignore_slots:
            return super().__new__(cls, name, bases, namespace, **kwargs)

        # dict.fromkeys works as an OrderedSet
        abases = dict.fromkeys(b for base in bases for b in base.__mro__)
        # Remove useless base classes
        for clsb in NamedMutableSequence.__mro__:
            del abases[clsb]
        # Get annotations in reverse mro order for the variable names
        types = reduce(
            lambda x, y: {**x, **y},
            (base.__annotations__ for base in reversed(abases)),
            cast(Dict[str, Any], {})
        )
        types.update(namespace.get('__annotations__', {}))
        # Finally add the variable names
        namespace['__slots__'] = tuple(types.keys())
        return super().__new__(cls, name, bases, namespace, **kwargs)


class NamedMutableSequence(Sequence[T_co], ABC, ignore_slots=True, metaclass=NamedMutableSequenceMeta):
    __slots__: Tuple[str, ...] = ()
    __annotations__ = {}  # type: ignore[var-annotated]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        for k in self.__slots__:
            self.__setattr__(k, kwargs.get(k))

        if args:
            for k, v in zip(self.__slots__, args):
                self.__setattr__(k, v)

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        values = ', '.join('%s=%r' % (k, self.__getattribute__(k)) for k in self.__slots__)
        return '%s(%s)' % (clsname, values)

    __repr__ = __str__

    @overload
    def __getitem__(self, index: int) -> T_co:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[T_co, ...]:
        ...

    def __getitem__(self, index: int | slice) -> T_co | Tuple[T_co, ...]:
        if isinstance(index, slice):
            return tuple(
                self.__getattribute__(self.__slots__[i])
                for i in range(index.start, index.stop)
            )
        return self.__getattribute__(self.__slots__[index])

    def __setitem__(self, item: int, value: Any) -> None:
        self.__setattr__(self.__slots__[item], value)

    def __len__(self) -> int:
        return self.__slots__.__len__()


class SliceableDeque(Deque[T]):

    def _resolve_slice(self, s: slice) -> slice:
        start = 0 if s.start is None else s.start if s.start >= 0 else len(self) + s.start
        stop = len(self) if s.stop is None else s.stop if s.stop >= 0 else len(self) + s.stop
        return slice(start, stop, s.step)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Deque[T]:
        ...

    def __getitem__(self, index: SupportsIndex | slice) -> T | Deque[T]:
        if isinstance(index, SupportsIndex):
            return super().__getitem__(index)
        index = self._resolve_slice(index)
        self.rotate(- index.start)
        sliced = islice(self.copy(), 0, index.stop - index.start, index.step)
        self.rotate(index.start)
        return deque(sliced)

    def __setitem__(self, index: int, value: T) -> None:
        return super().__setitem__(index, value)
        try:
            return super().__setitem__(index, value)
        except ValueError:
            index = self._resolve_slice(index)
            for i, v in zip(range(index.start, index.stop, index.step), value):
                self.rotate(- i)
                self.popleft()
                self.appendleft(v)
                self.rotate(i)
        if isinstance(index, SupportsIndex) and not isinstance(value, Iterable):
            return super().__setitem__(index, value)
        elif isinstance(index, slice) and isinstance(value, Iterable):
            index = self._resolve_slice(index)
            for i, v in zip(range(index.start, index.stop, index.step), value):
                self.rotate(- i)
                self.popleft()
                self.appendleft(v)
                self.rotate(i)
        elif isinstance(index, SupportsIndex) and not isinstance(value, Iterable):
            raise TypeError(f'{self.__class__.__name__}: can only assign a value!')
        elif isinstance(index, slice) and isinstance(value, Iterable):
            raise TypeError(f'{self.__class__.__name__}: can only assign an iterable!')
        else:
            raise NotImplementedError(f'{self.__class__.__name__}: not supported')

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, int):
            return super().__delitem__(index)
        for i, idx in enumerate(range(index.start, index.stop, index.step)):
            self.rotate(- idx + i)
            self.popleft()
            self.rotate(idx - i)
