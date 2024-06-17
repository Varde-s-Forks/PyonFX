from __future__ import annotations

__all__ = ['Time']

import math
import warnings

from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat
else:
    ConvertibleToFloat = Any

from ._logging import logger
from .misc import cround


class Time(float):
    """Time interface"""
    def __new__(cls, seconds: ConvertibleToFloat) -> Time:
        """
        Make a new Time object

        :param seconds:         Seconds
        :return:                Time object
        """
        return super().__new__(cls, seconds)

    def __add__(self: Self, __x: float) -> Self:
        return self.__class__(super().__add__(float(__x)))

    def __sub__(self: Self, __x: float) -> Self:
        return self.__class__(super().__sub__(float(__x)))

    def __mul__(self: Self, __x: float) -> Self:
        return self.__class__(super().__mul__(float(__x)))

    def __floordiv__(self: Self, __x: float) -> Self:
        return self.__class__(super().__floordiv__(float(__x)))

    def __truediv__(self: Self, __x: float) -> Self:
        return self.__class__(super().__truediv__(float(__x)))

    def __mod__(self: Self, __x: float) -> Self:
        return self.__class__(super().__mod__(float(__x)))

    def __radd__(self: Self, __x: float) -> Self:
        return self.__class__(super().__radd__(float(__x)))

    def __rsub__(self: Self, __x: float) -> Self:
        return self.__class__(super().__rsub__(float(__x)))

    def __rmul__(self: Self, __x: float) -> Self:
        return self.__class__(super().__rmul__(float(__x)))

    def __rfloordiv__(self: Self, __x: float) -> Self:
        return self.__class__(super().__rfloordiv__(float(__x)))

    def __rtruediv__(self: Self, __x: float) -> Self:
        return self.__class__(super().__rtruediv__(float(__x)))

    def __rmod__(self: Self, __x: float) -> Self:
        return self.__class__(super().__rmod__(float(__x)))

    def __neg__(self: Self) -> Self:
        return self.__class__(super().__neg__())

    def __pos__(self: Self) -> Self:
        return self.__class__(super().__pos__())

    def ts(self, precision: Literal[0, 3, 6, 9] = 3) -> str:
        """
        Get the Time object as a timestamp in the form of hh:mmm:ss.ms

        :param precision:       Precision of the seconds.
                                Possible values are 0, 3, 6 or 9, defaults to 3
        :return:                Timestamp.
        """
        m = self // 60
        s = self % 60
        h = m // 60
        m %= 60
        return composets(h, m, s, precision=precision)

    def assts(self, fps: Fraction | float, /, is_start: bool) -> str:
        """
        Get the Time object as a ASS timestamp in the form of hh:mmm:ss.ms

        :param fps:             Framerate Per Second.
        :param is_start:        Whether the time is a start time or not.
        :return:                ASS timestamp.
        """
        s = self - fps ** -1 * 0.5
        s = bound2assframe(s, fps, is_start, shifted=True)
        s = min(max(Time(0), s), Time(35999.999))
        ts = s.ts(precision=3)
        return ts[1:-1]

    def frame(self, fps: Fraction | float) -> int:
        """
        Get the current frame of the current Time.

        :param fps:             Framerate Per Second.
        :return:                Frame.
        """
        return round(self * fps)

    def assframe(self, fps: Fraction | float, is_start: bool) -> int:
        """
        Get the current ASS frame of the current Time

        :param fps:             Framerate Per Second.
        :param is_start:        Whether the time is a start time or not.
        :return:                ASS frame.
        """
        return math.ceil(self.__float__() * fps) - (0 if is_start else 1)

    def ass_frame(self, fps: Fraction | float, is_start: bool) -> int:
        warnings.warn('Deprecated method; use "assframe" instead')
        return self.assframe(fps, is_start)

    @classmethod
    def from_ts(cls, ts: str, /) -> Time:
        """
        Make a new Time object from a timestamp

        :param ts:              Timestamp
        :return:                Time object
        """
        h, m, s = map(float, ts.split(':'))
        return cls(h * 3600 + m * 60 + s)

    @classmethod
    def from_assts(cls, assts: str, /, fps: Fraction | float, is_start: bool) -> Time:
        """
        Make a new Time object from an ASS timestamp

        :param assts:           ASS timestamp.
        :param fps:             Framerate Per Second.
        :param is_start:        Whether the time is a start time or not.
        :return:                Time object
        """
        time = cls.from_ts(assts)
        if time == 0.0:
            return time

        t = bound2assframe(time, fps, is_start, shifted=True)
        t += fps ** -1 * 0.5
        return t

    @classmethod
    def from_frame(cls, f: int, fps: Fraction | float, /) -> Time:
        """
        Make a new Time object from a frame number

        :param f:               Frame number
        :param fps:             Framerate Per Second.
        :return:                Time object
        """
        if f == 0:
            return cls(0.0)

        t = round(float(10 ** 9 * f * fps ** -1))
        return cls(t / 10 ** 9)

    @classmethod
    def from_assframe(cls, f: int, fps: Fraction | float, /, is_start: bool) -> Time:
        """
        Make a new Time object from an ASS frame number

        :param f:               ASS frame number
        :param fps:             Framerate Per Second.
        :param is_start:        Whether the time is a start time or not.
        :return:                Time object
        """
        if is_start and f == 0:
            return cls(0.0)

        curr_ms = cround(Time.from_frame(f, fps) * 1000 + 1e-6)
        if is_start:
            prev_ms = cround(Time.from_frame(f - 1, fps) * 1000 + 1e-6)
            ms = prev_ms + int((curr_ms - prev_ms + 1) / 2)
        else:
            next_ms = cround(Time.from_frame(f + 1, fps) * 1000 + 1e-6)
            ms = curr_ms + int((next_ms - curr_ms + 1) / 2)
        s = ms / 1000
        return cls(s)


@lru_cache(maxsize=None)
def bound2frame(time: Time, fps: Fraction, /) -> Time:
    """
    Bound the time to its current displayed frame.

    :param time:            Time object
    :param fps:             Framerate Per Second.
    :return:                New Time object
    """
    return Time.from_frame(time.frame(fps), fps)


@lru_cache(maxsize=None)
def bound2assframe(time: Time, fps: Fraction | float, /, is_start: bool, shifted: bool = False) -> Time:
    """
    Bound the time to its current displayed frame in Aegisub

    :param time:            Time object
    :param fps:             Framerate Per Second.
    :param is_start:        Whether the time is a start time or not.
    :param shifted:         Whether the time has already been shifted from Aegisub or not, defaults to False
    :return:                New Time object
    """
    if time == 0.0:
        return time

    if not shifted:
        time -= fps ** -1 * 0.5

    f = time.assframe(fps, is_start)
    ntime = Time.from_assframe(f, fps, is_start)

    if not shifted:
        ntime += fps ** -1 * 0.5
    return ntime


@logger.catch(force_exit=True)
def composets(h: float, m: float, s: float, /, *, precision: Literal[0, 3, 6, 9] = 3) -> str:
    """
    :param h:           Hours.
    :param m:           Minutes.
    :param s:           Seconds.
    :param precision:   Precision of the seconds, defaults to 3.
    :return:            Timestamp.
    """
    if precision == 0:
        out = f"{h:02.0f}:{m:02.0f}:{round(s):02}"
    elif precision == 3:
        out = f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
    elif precision == 6:
        out = f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
    elif precision == 9:
        out = f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
    else:
        raise ValueError(f'composets: the precision {precision} must be a multiple of 3 (including 0)')
    return out
