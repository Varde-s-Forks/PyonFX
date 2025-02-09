"""Base coordinates system interface"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from math import ceil, floor, trunc
from typing import Any, Callable, NoReturn, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from .._logging import logger
from ..ptypes import NamedMutableSequence, SomeArrayLike

_CT = TypeVar('_CT', bound='Coordinates')


class Coordinates(NamedMutableSequence[float], ABC, empty_slots=True):
    """Abstract coordinate system"""

    @logger.catch
    def __contains__(self, o: object) -> NoReturn:
        raise NotImplementedError

    @property
    def __self_proxy(self) -> Coordinates:
        return self

    def __add__(self: _CT, _p: tuple[float, ...]) -> _CT:
        return self.__class__(*[sum(r, 0.0) for r in zip(self.__self_proxy, _p)])

    def __radd__(self: _CT, _p: tuple[float, ...]) -> _CT:
        return self.__add__(_p)

    def __sub__(self: _CT, _p: tuple[float, ...]) -> _CT:
        return self.__class__(*[c0 - c1 for (c0, c1) in zip(self.__self_proxy, _p)])

    def __rsub__(self: _CT, _p: tuple[float, ...]) -> _CT:
        return self.__class__(*[c1 - c0 for (c0, c1) in zip(self.__self_proxy, _p)])

    def __mul__(self: _CT, _p: int | tuple[float, ...]) -> _CT:
        if isinstance(_p, int):
            nattrs = [getattr(self, attr) * _p for attr in self.__slots__]
        else:
            nattrs = [c0 * c1 for (c0, c1) in zip(self.__self_proxy, _p)]
        return self.__class__(*nattrs)

    def __rmul__(self: _CT, _p: int | tuple[float, ...]) -> _CT:
        return self.__mul__(_p)

    def __matmul__(self: _CT, _mat: SomeArrayLike) -> _CT:
        return self.__class__(*_get_matmul_func(self.__self_proxy[:len(_mat)], _mat))

    def __rmatmul__(self: _CT, _mat: SomeArrayLike) -> _CT:
        return self.__class__(*_get_matmul_func(_mat, self.__self_proxy[:len(_mat)]))

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> NDArray[Any]:
        return np.array(tuple(self), dtype, copy=copy)

    def __neg__(self: _CT) -> _CT:
        return self.__class__(*[- a for a in self])

    def __pos__(self: _CT) -> _CT:
        return self.__class__(*[+ a for a in self])

    def __abs__(self: _CT) -> _CT:
        return self.__class__(*[abs(a) for a in self])

    def __setattr_iter(self, func: Callable[[float], int | float]) -> None:
        for attr, value in zip(self.__slots__, (func(x) for x in self)):
            setattr(self, attr, value)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Coordinates):
            return NotImplemented
        return __o._asdict() == self._asdict()

    def round(self, ndigits: int = 3) -> None:
        """
        Round the Point to a given precision in decimal digits.

        :param ndigits:         Number of digits, defaults to None
        """
        self.__setattr_iter(lambda x: round(x, ndigits))

    def trunc(self) -> None:
        """
        Truncates the Point to the nearest integer toward 0.
        """
        self.__setattr_iter(trunc)

    def floor(self) -> None:
        """
        Return the floor of the Point as an integer.

        :return:                New floored Point
        """
        self.__setattr_iter(floor)

    def ceil(self) -> None:
        """
        Return the ceiling of the Point as an integer.
        """
        self.__setattr_iter(ceil)

    @abstractmethod
    def to_2d(self) -> Any:
        """
        Convert current coordinate in the 2D cartesian system.
        If the current system has more axis than the output one, the conversion will be lossy,
        meaning that the last axis will just be ignored.
        """
        ...

    @abstractmethod
    def to_3d(self) -> Any:
        """
        Convert current coordinate in the 3D cartesian system.
        If the current system has less axis than the output one, the z axis will be set to 0
        """
        ...

    @abstractmethod
    def to_polar(self) -> Any:
        """
        Convert current coordinate to 2D polar system.
        If the current system has more axis than the output one, the conversion will be lossy,
        meaning that the last axis will just be ignored.
        """
        ...

    @abstractmethod
    def to_cylindrical(self) -> Any:
        """
        Convert current coordinate in the polar cylindrical system.
        If the current system has less axis than the output one, the z axis will be set to 0
        """
        ...

    @abstractmethod
    def to_spherical(self) -> Any:
        """
        Convert current coordinate in the polar spherical system.
        If the current system has less axis than the output one, the theta axis will be set to 90 degrees
        """
        ...


class Axis(IntEnum):
    """Base axis enum"""
    ...


def _get_matmul_func(_mat1: SomeArrayLike, _mat2: SomeArrayLike) -> map[float]:
    return map(float, np.asarray(_mat1) @ np.asarray(_mat2))
