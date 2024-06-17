"""Internal types module"""
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from functools import _lru_cache_wrapper, wraps
from os import PathLike
from types import FunctionType, MemberDescriptorType, MethodType
from typing import (
    AbstractSet, Any, Callable, Collection, Dict, Generic, Iterable, Iterator, Literal,
    MutableMapping, MutableSet, Reversible, Sequence, Tuple, TypeVar, Union, cast, final, get_args,
    get_origin, overload
)

from numpy.typing import NDArray
from typing_extensions import Annotated, get_type_hints

T = TypeVar('T')
_T = TypeVar('_T')
S = TypeVar('S')
T_co = TypeVar('T_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
TCV_co = TypeVar('TCV_co', bound=Union[float, int, str], covariant=True)  # Type Color Value covariant
TCV_inv = TypeVar('TCV_inv', bound=Union[float, int, str])  # Type Color Value invariant
ACV = Union[float, int, str]
Nb = TypeVar('Nb', bound=Union[float, int])  # Number
Tup3 = Tuple[Nb, Nb, Nb]
Tup4 = Tuple[Nb, Nb, Nb, Nb]
Tup3Str = Tuple[str, str, str]
AnyPath = Union[PathLike[str], str]
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
        val = [val] if isinstance(val, (int, float)) else val
        for v in val:
            if not self.x < v <= self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


class ValueRangeIncInc(CheckAnnotated[float]):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def check(self, val: float | Iterable[float], param_name: str) -> None:
        val = [val] if isinstance(val, (int, float)) else val
        for v in val:
            if not self.x < v < self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


Nb8bit = Annotated[int, ValueRangeInclExcl(0, 256)]
Nb16bit = Annotated[int, ValueRangeInclExcl(0, 65536)]
NbFloat = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Pct = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Alignment = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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


class AutoSlotsMeta(ABCMeta):
    @classmethod
    def __prepare__(cls, __name: str, __bases: Tuple[type, ...], **kwargs: Any) -> MutableMapping[str, object]:
        return {'__slots__': (), '__slots_ex__': ()}

    def __new__(
        cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any],
        empty_slots: bool = False, slots_ex: bool = False,
        slots_ex_exclude: str | tuple[str, ...] | None = None,
        **kwargs: Any
    ) -> AutoSlotsMeta:
        if empty_slots:
            return super().__new__(cls, name, bases, namespace, **kwargs)

        # Get all base types in reverse mro
        abases = tuple(reversed(OrderedSet(b for base in bases for b in base.__mro__)))

        # Get all possible values to put in __slots__
        _slots_inherited = OrderedSet(
            banno
            for base in abases
            if hasattr(base, '__annotations__')
            for banno in base.__annotations__
        )

        # __annotations__ and __slots__ from the current class
        _slots: OrderedSet[str] = OrderedSet(namespace['__slots__'])
        _slots.update(namespace.get('__annotations__', {}))
        _all_slots = _slots_inherited | _slots

        # Get possible class variables & properties
        attrs = {
            attr: getattr(abase, attr) for abase in abases for attr in dir(abase)
        }
        attrs.update(namespace)
        attrs = {
            k: v for k, v in attrs.items()
            if not k.startswith('__') and not k.endswith('__')
            and not isinstance(v, (FunctionType, classmethod, staticmethod, MethodType, MemberDescriptorType, _lru_cache_wrapper))
            and k not in {'_abc_impl', '_is_protocol'}
        }

        namespace = {**attrs, **namespace}
        namespace['__slots__'] = tuple(k for k in _all_slots if k not in namespace)

        if slots_ex:
            sex_exld = (slots_ex_exclude, ) if isinstance(slots_ex_exclude, str) else slots_ex_exclude if slots_ex_exclude else tuple[str]()
            namespace['__slots_ex__'] = namespace['__slots__'] + tuple(set(attrs) - set(sex_exld))

        return super().__new__(cls, name, bases, namespace, **kwargs)


class AutoSlots(ABC, empty_slots=True, metaclass=AutoSlotsMeta):
    __slots__: Tuple[str, ...]
    __slots_ex__: Tuple[str, ...]

    @property
    def __all_slots__(self) -> Tuple[str, ...]:
        return tuple(OrderedSet(self.__slots__ + self.__slots_ex__))

    def __delattrs__(self) -> None:
        for k in self.__all_slots__:
            try:
                super().__delattr__(k)
            except AttributeError:
                pass


class NamedMutableSequence(AutoSlots, Sequence[T_co], Generic[T_co], ABC, empty_slots=True):
    def __init__(self, *args: T_co, **kwargs: T_co) -> None:
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        if args:
            for k, v in zip(self.__slots__, args):
                self.__setattr__(k, v)

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        values = ', '.join('%s=%r' % (k, self.__getattribute__(k)) for k in self.__slots__)
        return '%s(%s)' % (clsname, values)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return NotImplemented
        return type(self) is type(__o) and tuple(self) is tuple(__o)

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

    def _asdict(self) -> Dict[str, T_co]:
        return {k: v for k, v in zip(self.__slots__, self)}


class OrderedSet(MutableSet[T], Generic[T], ABC):
    __slots__ = '__odict'
    __odict: OrderedDict[T, Any | None]

    def __init__(self, __iterable: Iterable[T] | None = None, /) -> None:
        if __iterable is not None:
            self.__odict = OrderedDict.fromkeys(__iterable, None)
        else:
            self.__odict = OrderedDict()

    def __str__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, ', '.join(str(v) for v in self))

    def __repr__(self) -> str:
        return self.__str__()

    def __reversed__(self) -> Iterator[T]:
        return reversed(self.__odict)

    # Abstract methods
    def __contains__(self, x: object) -> bool:
        return x in self.__odict

    def __iter__(self) -> Iterator[T]:
        return self.__odict.__iter__()

    def __len__(self) -> int:
        return self.__odict.__len__()

    def add(self, __element: T, /) -> None:
        """
        Add an element to a set.

        This has no effect if the element is already present.

        :param __element:       Element to add
        """
        self.__odict[__element] = None

    def discard(self, __element: T, /) -> None:
        """
        Remove an element from a set if it is a member.

        If the element is not a member, do nothing.

        :param __element:       Element to remove
        """
        try:
            del self.__odict[__element]
        except KeyError:
            pass

    # Redefining methods for return types because they're just wrong
    def __and__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__and__(s)  # type: ignore[return-value]

    def __iand__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__iand__(s)  # type: ignore[return-value]

    def __or__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__or__(s)  # type: ignore[return-value]

    def __ior__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__ior__(s)  # type: ignore

    def __sub__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__sub__(s)  # type: ignore[return-value]

    def __isub__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__isub__(s)  # type: ignore[return-value]

    def __xor__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__xor__(s)  # type: ignore[return-value]

    def __ixor__(self, s: AbstractSet[_T]) -> OrderedSet[_T | T]:
        return super().__ixor__(s)  # type: ignore

    # Set methods
    def copy(self) -> OrderedSet[T]:
        """
        Return a shallow copy of a set
        """
        return OrderedSet(self.__odict.keys())

    def difference(self, *s: Iterable[S]) -> OrderedSet[S | T]:
        """
        Return the difference of two or more sets as a new set.

        (i.e. all elements that are in this set but not the others.)

        :param s:               Positional argument of Iterables
        :return:                OrderedSet of differences
        """
        return self - set(el for it in s for el in it)

    def difference_update(self, *s: Iterable[Any]) -> None:
        """
        Remove all elements of another set from this set.

        :param s:               Positional argument of Iterables
        """
        for el in set(el for it in s for el in it):
            self.discard(el)

    def intersection(self, *s: Iterable[S]) -> OrderedSet[S | T]:
        """
        Return the intersection of two sets as a new set.

        (i.e. all elements that are in both sets.)

        :param s:               Positional argument of Iterables
        :return:                OrderedSet of intersections
        """
        return self & set(el for it in s for el in it)

    def intersection_update(self, *s: Iterable[Any]) -> None:
        """
        Update a set with the intersection of itself and another.

        :param s:               Positional argument of Iterables
        """
        other = set(el for it in s for el in it)
        for element in self.__odict.copy():
            if element not in other:
                self.discard(element)

    def symmetric_difference(self, __s: Iterable[T], /) -> OrderedSet[T]:
        """
        Return the symmetric difference of two sets as a new set.

        (i.e. all elements that are in exactly one of the sets.)

        :param s:               An Iterable
        :return:                OrderedSet of symmetric differences
        """
        return self ^ set(__s)

    def symmetric_difference_update(self, __s: Iterable[T], /) -> None:
        """
        Update a set with the symmetric difference of itself and another.

        :param __s:             An Iterable
        """
        other = set(__s)
        for element in self.__odict.copy():
            if element in self.__odict and element in other:
                self.remove(element)

    def union(self, *s: Iterable[S]) -> OrderedSet[S | T]:
        """
        Return the union of sets as a new set.

        (i.e. all elements that are in either set.)

        :param s:               Positional argument of Iterables
        :return:                OrderedSet of unions
        """
        return self | set(el for it in s for el in it)

    def update(self, *s: Iterable[T]) -> None:
        """
        Update a set with the union of itself and others.

        :param s:               Positional argument of Iterables
        """
        self.__odict.update((el, None) for it in s for el in it)


_CustomBoolT = TypeVar('_CustomBoolT', bound='CustomBool')


class CustomBool(int):
    def __copy__(self: _CustomBoolT) -> _CustomBoolT:
        return self.__class__(self.__repr__())

    def __deepcopy__(self: _CustomBoolT, *args: Any) -> _CustomBoolT:
        return self.__copy__()


@final
class AssBool(CustomBool):
    def __new__(cls, __o: str = 'no') -> AssBool:
        return super().__new__(cls, bool(__o == 'yes'))

    def __str__(self) -> str:
        return {1: 'True', 0: 'False'}[self]

    def __repr__(self) -> str:
        return {1: 'yes', 0: 'no'}[self]


@final
class StyleBool(CustomBool):
    def __new__(cls, __o: int | str = 0) -> StyleBool:
        return super().__new__(cls, __o)

    def __str__(self) -> str:
        return {-1: 'True', 0: 'False'}[self]


@final
class BorderStyleBool(CustomBool):
    def __new__(cls, __o: int | str = 1) -> BorderStyleBool:
        return super().__new__(cls, __o)

    def __str__(self) -> str:
        return {3: 'True', 1: 'False'}[self]
