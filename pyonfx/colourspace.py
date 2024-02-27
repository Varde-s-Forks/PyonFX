"""Colourspace module"""
from __future__ import annotations

__all__ = [
    'ColourSpace',
    'RGBS', 'RGBAS',
    'RGB', 'RGB24', 'RGB30', 'RGB36', 'RGB42', 'RGB48',
    'RGBA', 'RGBA32', 'RGBA40', 'RGBA48', 'RGBA56', 'RGBA64',
    'HSL', 'HSV',
    'HTML', 'ASSColor', 'Opacity',
    'XYZ', 'xyY', 'Lab', 'LCHab', 'Luv', 'LCHuv'
]

import re
from abc import ABC, abstractmethod
from typing import Any, Tuple, Type, TypeVar, cast, overload

from typing_extensions import TypeGuard

from ._logging import logger
from .convert import ConvertColour as CC
from .misc import clamp_value
from .ptypes import ACV, NamedMutableSequence, Nb, Nb8bit, Pct, TCV_co, Tup4

_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

_ColourSpaceT = TypeVar('_ColourSpaceT', bound='ColourSpace[TCV_co]')  # type: ignore
_NumBasedT = TypeVar('_NumBasedT', bound='_NumBased[Nb]')  # type: ignore
_RGB_T = TypeVar('_RGB_T', bound='_BaseRGB[Nb]')  # type: ignore
_HueSaturationBasedT = TypeVar('_HueSaturationBasedT', bound='_HueSaturationBased')
_OpacityT = TypeVar('_OpacityT', bound='Opacity')


class ColourSpace(NamedMutableSequence[TCV_co], ABC, empty_slots=True):
    """Base class for colourspace interface"""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __copy__(self: _ColourSpaceT) -> _ColourSpaceT:
        vals = tuple(getattr(self, x) for x in self.__slots__ if not x.startswith('_'))
        return self.__class__(vals[0] if len(vals) <= 1 else vals)

    def __deepcopy__(self: _ColourSpaceT, *args: Any) -> _ColourSpaceT:
        return self.__copy__()

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        values = ', '.join('%s=%r' % (k, self.__getattribute__(k)) for k in self.__slots__ if not k.startswith('_'))
        return '%s(%s)' % (clsname, values)

    def __repr__(self) -> str:
        return super().__str__()

    @abstractmethod
    def interpolate(self: _ColourSpaceT, nobj: _ColourSpaceT, pct: Pct, /) -> _ColourSpaceT:
        """
        Interpolate the colour values of the current object with nobj
        and returns a new interpolated object.

        :param nobj:            Second colourspace. Must be of the same type
        :param pct:             Percentage value in the range 0.0 - 1.0
        :return:                New colourspace object
        """
        ...

    @abstractmethod
    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        """
        Convert current object to an RGB type object

        :param rgb_type:    RGB type
        :return:            RGB object
        """
        ...

    @abstractmethod
    def to_xyz(self) -> XYZ:
        """
        Convert current object to a XYZ object

        :return:            XYZ object
        """
        ...

    @abstractmethod
    def to_xyy(self) -> xyY:
        """
        Convert current object to a xyY object

        :return:            xyY object
        """
        ...

    @abstractmethod
    def to_lab(self) -> Lab:
        """
        Convert current object to a Lab object

        :return:            Lab object
        """
        ...

    @abstractmethod
    def to_lch_ab(self) -> LCHab:
        """
        Convert current object to a LCHab object

        :return:            LCHab object
        """
        ...

    @abstractmethod
    def to_luv(self) -> Luv:
        """
        Convert current object to a Luv object

        :return:            Luv object
        """
        ...

    @abstractmethod
    def to_lch_uv(self) -> LCHuv:
        """
        Convert current object to a LCHuv object

        :return:            LCHuv object
        """
        ...

    @abstractmethod
    def to_hsl(self) -> HSL:
        """
        Convert current object to a HSL object

        :return:            HSL object
        """
        ...

    @abstractmethod
    def to_hsv(self) -> HSV:
        """
        Convert current object to a HSV object

        :return:            HSV object
        """
        ...

    @abstractmethod
    def to_html(self) -> HTML:
        """
        Convert current object to a HTML object

        :return:            HTML object
        """
        ...

    @abstractmethod
    def to_ass_color(self) -> ASSColor:
        """
        Convert current object to an ASSColor object

        :return:            AssColor object
        """
        ...


class _NumBased(ColourSpace[Nb], ABC, empty_slots=True):
    """Number based colourspace"""

    @logger.catch
    def interpolate(self: _NumBasedT, nobj: _NumBasedT, pct: Pct, /) -> _NumBasedT:
        if not isinstance(nobj, self.__class__):
            raise ValueError(f'{self.__class__.__name__}: {nobj} is not of the same type')
        return self.__class__(tuple(  # type: ignore[var-annotated]
            (1 - pct) * cs1_val + pct * cs2_val
            for cs1_val, cs2_val in zip(self, nobj)
        ))


class _ForceNumber(_NumBased[Nb], ABC, empty_slots=True):
    """Base class for clamping and forcing type values"""

    peaks: Tuple[Nb, Nb]
    """Max value allowed"""

    force_type: Type[Nb]
    """Forcing type"""

    # TODO: Maybe yeet __setattr__ and __delattr__ since they're very slow
    @logger.catch
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'peaks', 'force_type'}:
            raise ValueError(f'{self.__class__.__name__}: Can\'t change {name}')
        if not name.startswith('_'):
            value = clamp_value(
                self.force_type(value),
                self.force_type(self.peaks[0]),
                self.force_type(self.peaks[1])
            )
        super().__setattr__(name, value)

    @logger.catch
    def __delattr__(self, name: str) -> None:
        if name in {'peaks', 'force_type'}:
            raise ValueError(f'{self.__class__.__name__}: Can\'t delete {name}')
        return super().__delattr__(name)


class _ForceFloat(_ForceNumber[float], ABC, empty_slots=True):
    """Force values to float and clamp in the range peaks"""

    force_type: Type[float] = float

    def round(self, ndigits: int) -> None:
        """
        Round a number to a given precision in decimal digits.

        :param ndigits:         Number of digits
        """
        for attr, value in zip(self.__slots__, self):
            setattr(self, attr, round(value, ndigits))


class _ForceInt(_ForceNumber[int], ABC, empty_slots=True):
    """Force values to int (truncate them if necessary) and clamp in the range peaks"""
    force_type: Type[int] = int


class _BaseRGB(ColourSpace[Nb], ABC, empty_slots=True):
    """Base class for RGB colourspaces"""
    r: Nb
    """Red value"""
    g: Nb
    """Green value"""
    b: Nb
    """Blue value"""

    peaks: Tuple[Nb, Nb]
    """Max value allowed"""

    def __new__(cls: Type[_RGB_T], _x: ColourSpace[TCV_co] | Tuple[Nb, ...]) -> _RGB_T:
        return _x.to_rgb(cls) if not isinstance(_x, tuple) else super().__new__(cls)

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[Nb, ...]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.r, self.g, self.b = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        if type(self) is rgb_type:
            return self.__copy__()  # type: ignore

        newpeaks = rgb_type.peaks

        nvalues = tuple((1 / self.peaks[1]) * val for val in self)
        svalues = tuple(v * newpeaks[1] for v in nvalues)

        if newpeaks[1] != 1:
            svalues = tuple(round(sval) for sval in svalues)

        return rgb_type(svalues)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.rgb_to_xyz(*self.to_rgb(RGBS)))

    def to_xyy(self) -> xyY:
        return xyY(CC.rgb_to_xyy(*self.to_rgb(RGBS)))

    def to_lab(self) -> Lab:
        return Lab(CC.rgb_to_lab(*self.to_rgb(RGBS)))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.rgb_to_lch_ab(*self.to_rgb(RGBS)))

    def to_luv(self) -> Luv:
        return Luv(CC.rgb_to_luv(*self.to_rgb(RGBS)))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.rgb_to_lch_uv(*self.to_rgb(RGBS)))

    def to_hsl(self) -> HSL:
        return HSL(CC.rgb_to_hsl(*self.to_rgb(RGBS)))

    def to_hsv(self) -> HSV:
        return HSV(CC.rgb_to_hsv(*self.to_rgb(RGBS)))

    def to_html(self) -> HTML:
        r, g, b = self.to_rgb(RGB)
        return HTML((r, g, b))

    def to_ass_color(self) -> ASSColor:
        return ASSColor(
            '&H'
            + ''.join(hex(x)[2:].zfill(2) for x in reversed(self.to_rgb(RGB)))
            + '&'
        )


class _RGBNoAlpha(_BaseRGB[Nb], ABC, empty_slots=True):
    """Base class for RGB colourspaces without alpha"""

    def __new__(cls, _x: ColourSpace[ACV] | Tuple[Nb, Nb, Nb]) -> _RGBNoAlpha[Nb]:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object or tuple of three numbers R, G and B values
        """
        return super().__new__(cls, _x)  # type: ignore[arg-type]

    def __init__(self, _x: ColourSpace[ACV] | Tuple[Nb, Nb, Nb]) -> None:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object or tuple of three numbers R, G and B values
        """
        super().__init__(_x)


class _RGBAlpha(_BaseRGB[Nb], ABC, empty_slots=True):
    """Base class for RGB colourspaces with alpha"""

    a: Nb
    """Alpha value"""

    def __new__(cls, _x: ColourSpace[ACV] | Tuple[Nb, Nb, Nb] | Tuple[Nb, Nb, Nb, Nb]) -> _RGBAlpha[Nb]:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object
                            or tuple of three numbers R, G and B values
                            or tuple of four numbers R, G, B and Alpha values
        """
        return super().__new__(cls, _x)  # type: ignore[arg-type]

    def __init__(self, _x: ColourSpace[ACV] | Tuple[Nb, Nb, Nb] | Tuple[Nb, Nb, Nb, Nb]) -> None:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object
                            or tuple of three numbers R, G and B values
                            or tuple of four numbers R, G, B and Alpha values
        """
        if isinstance(_x, tuple):
            super().__init__(_x[:3])
            if len(_x) > 3:
                self.a = _x[-1]
            else:
                self.a = self.peaks[1]
        else:
            super().__init__(_x)



class RGBS(_RGBNoAlpha[float], _ForceFloat):
    """RGB colourspace in range 0.0 - 1.0"""

    peaks: Tuple[float, float] = (0., 1.)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__(_x)


class RGBAS(_RGBAlpha[float], _ForceFloat):
    """RGB with alpha colourspace in range 0.0 - 1.0"""

    peaks: Tuple[float, float] = (0., 1.)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float, float], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float] | Tuple[float, float, float, float]) -> None:
        super().__init__(_x)


class RGB(_RGBNoAlpha[int], _ForceInt):
    """RGB colourspace in range 0 - 255"""
    peaks: Tuple[int, int] = (0, (2 ** 8) - 1)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tuple[int, int, int], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[int, int, int]) -> None:
        super().__init__(_x)


class RGB24(RGB):
    """RGB colourspace in range 0 - 255"""
    ...


class RGB30(RGB, slots_ex=True):
    """RGB colourspace in range 0 - 1023"""
    peaks: Tuple[int, int] = (0, (2 ** 10) - 1)


class RGB36(RGB):
    """RGB colourspace in range 0 - 4095"""
    peaks: Tuple[int, int] = (0, (2 ** 12) - 1)


class RGB42(RGB):
    """RGB colourspace in range 0 - 16383"""
    peaks: Tuple[int, int] = (0, (2 ** 14) - 1)


class RGB48(RGB):
    """RGB colourspace in range 0 - 65535"""
    peaks: Tuple[int, int] = (0, (2 ** 16) - 1)


class RGBA(_RGBAlpha[int], _ForceInt):
    """RGB with alpha colourspace in range 0 - 255"""

    peaks: Tuple[int, int] = (0, (2 ** 8) - 1)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tuple[int, int, int], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup4[int], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[int, int, int] | Tup4[int]) -> None:
        super().__init__(_x)


class RGBA32(RGBA):
    """RGB with alpha colourspace in range 0 - 255"""
    ...


class RGBA40(RGBA):
    """RGB with alpha colourspace in range 0 - 1023"""
    peaks: Tuple[int, int] = (0, (2 ** 10) - 1)


class RGBA48(RGBA):
    """RGB with alpha colourspace in range 0 - 4095"""
    peaks: Tuple[int, int] = (0, (2 ** 12) - 1)


class RGBA56(RGBA):
    """RGB with alpha colourspace in range 0 - 16383"""
    peaks: Tuple[int, int] = (0, (2 ** 14) - 1)


class RGBA64(RGBA):
    """RGB with alpha colourspace in range 0 - 65535"""
    peaks: Tuple[int, int] = (0, (2 ** 16) - 1)


class _HueSaturationBased(_ForceFloat, ColourSpace[float], ABC, empty_slots=True):
    """Base class for Hue and Saturation based colourspace"""

    h: float
    """Hue value"""

    s: float
    """Saturation"""

    peaks: Tuple[float, float] = (0., 1.)

    @abstractmethod
    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()

    def to_xyz(self) -> XYZ:
        return self.to_rgb(RGBS).to_xyz()

    def to_xyy(self) -> xyY:
        return self.to_xyz().to_xyy()

    def to_lab(self) -> Lab:
        return self.to_xyz().to_lab()

    def to_lch_ab(self) -> LCHab:
        return self.to_xyz().to_lch_ab()

    def to_luv(self) -> Luv:
        return self.to_xyz().to_luv()

    def to_lch_uv(self) -> LCHuv:
        return self.to_xyz().to_lch_uv()

    def to_html(self) -> HTML:
        return self.to_rgb(RGB).to_html()

    def to_ass_color(self) -> ASSColor:
        return self.to_rgb(RGB).to_ass_color()

    @classmethod
    def from_ass_val(cls: Type[_HueSaturationBasedT], _x: Tuple[int, int, int]) -> _HueSaturationBasedT:
        """
        Make a Hue Saturation based object from ASS values

        :param _x:          Tuple of integer in the range 0 - 255
        :return:            Hue Saturation based object
        """
        return cls(cast(Tuple[float, float, float], tuple(x / 255 for x in _x)))

    def to_ass_val(self) -> Tuple[float, float, float]:
        """
        Make a tuple of float of this current object in range 0 - 255

        :return:            Tuple of integer in the range 0 - 255
        """
        return cast(Tuple[float, float, float], tuple(round(x * 255) for x in self))

    def as_chromatic_circle(self) -> Tuple[float, float, float]:
        """
        Change H in a chromatic circle in range 0.0 - 360.0

        :return:            Tuple of float with H in range 0.0 - 360.0
        """
        return cast(Tuple[float, float, float], (self.h * 360, *(*self, )[1:3]))


class HSL(_HueSaturationBased):
    """HSL colourspace in range 0.0 - 1.0"""

    l: float
    """Lightness value"""

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HSL:
        """
        Make a new HSL colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> HSL:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and L values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> HSL:
        return _x.to_hsl() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and L values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__(_x)
        if isinstance(_x, tuple):
            self.h, self.s, self.l = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.hsl_to_rgb(*self)).to_rgb(rgb_type)

    def to_hsl(self) -> HSL:
        return self

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()


class HSV(_HueSaturationBased):
    """HSV colourspace in range 0.0 - 1.0"""

    v: float
    """Value value"""

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HSV:
        """
        Make a new HSV colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> HSV:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and V values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> HSV:
        return _x.to_hsv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        """
        Make a new HSV colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float], /) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and V values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__(_x)
        if isinstance(_x, tuple):
            self.h, self.s, self.v = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.hsv_to_rgb(*self)).to_rgb(rgb_type)

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self


class Opacity(ColourSpace[float]):
    """Opacity colourspace like in range 0.0 - 1.0"""

    value: float
    """Value in floating format in the range 0.0 - 1.0"""

    def __init__(self, _x: Pct) -> None:
        """
        Make an Opacity colourspace object

        :param _x:      Percentage of the opacity.
                        1.0 means full opaque
                        0.0 means full transparent
        """
        super().__init__()
        self.value = clamp_value(_x, 0., 1.0)

    @property
    def ass_hex(self) -> str:
        """ASS value, hexadecimal inversed"""
        return f'&H{round(abs(self.value * 255 - 255)):02X}&'

    @overload
    @classmethod
    def from_ass_val(cls, _x: str) -> Opacity:
        """
        Convert an ASS string value to an Opacity object

        :param _x:          ASS alpha string in the format &HXX&
        :return:            Opacity object
        """
        ...

    @overload
    @classmethod
    def from_ass_val(cls, _x: Nb8bit) -> Opacity:
        """
        Convert an ASS integer value to an Opacity object

        :param _x:          ASS integer string in the range 0 - 255
        :return:            Opacity object
        """
        ...

    @classmethod
    @logger.catch(force_exit=True)
    def from_ass_val(cls, _x: str | Nb8bit) -> Opacity:
        if isinstance(_x, str):
            if not (fullmatch := re.fullmatch(r"&H([0-9A-F]{2})&", _x)):
                raise ValueError(f'Opacity: Provided ASS alpha string {_x} is not in the expected format &HXX&')
            x = float(int(fullmatch.group(1), 16))
        else:
            x = float(_x)
        x = (255 - x) / 255
        return cls(x)

    def interpolate(self: _OpacityT, nobj: _OpacityT, pct: Pct, /) -> _OpacityT:
        return self.__class__(self.value * (1 - pct) + nobj.value * pct)

    @logger.catch
    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        raise NotImplementedError

    @logger.catch
    def to_xyz(self) -> XYZ:
        raise NotImplementedError

    @logger.catch
    def to_xyy(self) -> xyY:
        raise NotImplementedError

    @logger.catch
    def to_lab(self) -> Lab:
        raise NotImplementedError

    @logger.catch
    def to_lch_ab(self) -> LCHab:
        raise NotImplementedError

    @logger.catch
    def to_luv(self) -> Luv:
        raise NotImplementedError

    @logger.catch
    def to_lch_uv(self) -> LCHuv:
        raise NotImplementedError

    @logger.catch
    def to_hsl(self) -> HSL:
        raise NotImplementedError

    @logger.catch
    def to_hsv(self) -> HSV:
        raise NotImplementedError

    @logger.catch
    def to_html(self) -> HTML:
        raise NotImplementedError

    @logger.catch
    def to_ass_color(self) -> ASSColor:
        raise NotImplementedError


class _HexBased(ColourSpace[str], ABC, empty_slots=True):
    """Hexadecimal based colourspace"""

    _rgb: RGB
    """Internal RGB object corresponding to the hexadecimal value"""

    data: str
    """Hexadecimal value"""

    def __copy__(self) -> _HexBased:
        return self.__class__(self._rgb)

    def interpolate(self, nobj: _ColourSpaceT, pct: Pct, /) -> _ColourSpaceT:
        return cast(_ColourSpaceT, self.__class__(self._rgb.interpolate(nobj.to_rgb(RGB), pct)))

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return self._rgb.to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return self.to_rgb(RGBS).to_xyz()

    def to_xyy(self) -> xyY:
        return self.to_xyz().to_xyy()

    def to_lab(self) -> Lab:
        return self.to_xyz().to_lab()

    def to_lch_ab(self) -> LCHab:
        return self.to_xyz().to_lch_ab()

    def to_luv(self) -> Luv:
        return self.to_xyz().to_luv()

    def to_lch_uv(self) -> LCHuv:
        return self.to_xyz().to_lch_uv()

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()

    @staticmethod
    def hex_to_int(h: str) -> int:
        """
        Convert hexadecimal to based 10 integer

        :param h:       Hexadecimal value
        :return:        Base 10 value
        """
        return int(h, 16)


def _istup3(tup: Tuple[_T1, _T1, _T1], t: Type[_T2]) -> TypeGuard[Tuple[_T2, _T2, _T2]]:
    return all(isinstance(x, t) for x in tup)


class HTML(_HexBased):
    """HTML colourspace object"""

    _rgb: RGB
    data: str

    @overload
    def __new__(cls, _x: str) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      HTML string
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[str, str, str]) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three hexadecimal values
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[int, int, int]) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three bases 10 values
        """
        ...

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Colourspace object
        """
        ...

    def __new__(cls, _x: str | Tuple[str, str, str] | Tuple[int, int, int] | ColourSpace[TCV_co]) -> HTML:
        return _x.to_html() if not isinstance(_x, (str, tuple)) else super().__new__(cls)

    @overload
    def __init__(self, _x: str) -> None:
        """
        Make a HTML colourspace object

        :param _x:      HTML string
        """
        ...

    @overload
    def __init__(self, _x: Tuple[str, str, str]) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three hexadecimal values
        """
        ...

    @overload
    def __init__(self, _x: Tuple[int, int, int]) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three bases 10 values
        """
        ...

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Colourspace object
        """
        ...

    @logger.catch
    def __init__(self, _x: str | Tuple[str, str, str] | Tuple[int, int, int] | ColourSpace[TCV_co]) -> None:
        super().__init__()

        if isinstance(_x, ColourSpace):
            return None

        if isinstance(_x, str):
            seq = _x
            fmatch = re.fullmatch(r"#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})", _x)
            if not fmatch:
                raise ValueError(f'{self.__class__.__name__}: No match found')
            # assert fmatch
            r, g, b = map(self.hex_to_int, fmatch.groups())
            self._rgb = RGB((r, g, b))
        elif _istup3(_x, int):
            self._rgb = RGB(_x)  # type: ignore[arg-type]
            seq = ''.join(hex(x)[2:].zfill(2) for x in self._rgb)
        elif _istup3(_x, str):
            r, g, b = map(self.hex_to_int, _x)  # type: ignore[arg-type]
            self._rgb = RGB((r, g, b))
            seq = ''.join(_x)  # type: ignore[arg-type]
        else:
            raise NotImplementedError

        self.data = "#" + seq.upper()

    def to_html(self) -> HTML:
        return self

    def to_ass_color(self) -> ASSColor:
        return self._rgb.to_ass_color()


class ASSColor(_HexBased):
    """AssColor colourspace object"""

    _rgb: RGB
    data: str

    @overload
    def __new__(cls, _x: str) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: Tuple[str, str, str]) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: Tuple[int, int, int]) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> ASSColor:
        ...

    def __new__(cls, _x: str | Tuple[str, str, str] | Tuple[int, int, int] | ColourSpace[TCV_co]) -> ASSColor:
        return _x.to_ass_color() if not isinstance(_x, (str, tuple)) else super().__new__(cls)

    @overload
    def __init__(self, _x: str) -> None:
        """
        Make a AssColor object from ASS string of the form "&HBBGGRR&"

        .. code-block:: python

            >>> ASSColor('&HF1D410&')

        :param _x:      ASS string
        """
        ...

    @overload
    def __init__(self, _x: Tuple[str, str, str]) -> None:
        """
        Make a AssColor object from a tuple of string of the form ('BB', 'GG', 'RR')

        .. code-block:: python

            >>> ASSColor(('F1', 'D4', '10'))

        :param _x:      Tuple of string
        """
        ...

    @overload
    def __init__(self, _x: Tuple[int, int, int]) -> None:
        """
        Make a AssColor object from a tuple of int of the form (BB, GG, RR)

        .. code-block:: python

            >>> ASSColor((241, 212, 16))
            >>> ASSColor((0xF1, 0xD4, 0x10))

        :param _x:      Tuple of integers
        """
        ...

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a AssColor object from an other ColourSpace object

        :param _x:      ColourSpace object
        """
        ...

    @logger.catch
    def __init__(self, _x: str | Tuple[str, str, str] | Tuple[int, int, int] | ColourSpace[TCV_co]) -> None:
        super().__init__()
        if isinstance(_x, ColourSpace):
            return None
        if isinstance(_x, str):
            if not (fmatch := re.fullmatch(r"&H([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})&", _x.upper())):
                raise ValueError(f'{self.__class__.__name__}: No match found')
            seq = _x[2:-1]
            r, g, b = map(self.hex_to_int, fmatch.groups()[::-1])
            self._rgb = RGB((r, g, b))
        elif _istup3(_x, int):
            self._rgb = RGB(_x[::-1])  # type: ignore[arg-type]
            seq = ''.join(hex(x)[2:].zfill(2) for x in _x)
        elif _istup3(_x, str):
            r, g, b = map(self.hex_to_int, _x)  # type: ignore[arg-type]
            self._rgb = RGB((r, g, b))
            seq = ''.join(_x)  # type: ignore[arg-type]
        else:
            raise NotImplementedError
        self.data = "&H" + seq.upper() + "&"

    def to_html(self) -> HTML:
        return self._rgb.to_html()

    def to_ass_color(self) -> ASSColor:
        return self


class XYZBased(_ForceFloat, ColourSpace[float], ABC):
    """Base colourspace class for colourspace where the conversions need XYZ"""

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()

    def to_html(self) -> HTML:
        return self.to_rgb(RGB).to_html()

    def to_ass_color(self) -> ASSColor:
        return self.to_rgb(RGB).to_ass_color()


class XYZ(XYZBased):
    """XYZ colourspace object"""

    x: float
    """Mix of the three CIE RGB curves chosen to be nonnegative value"""

    y: float
    """Luminance value"""

    z: float
    """Quasi-equal to blue value"""

    peaks: Tuple[float, float] = (0., 1.)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co], /) -> XYZ:
        """
        Make a XYZ colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float], /) -> XYZ:
        """
        Make a XYZ colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> XYZ:
        return _x.to_xyz() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        """
        Make a XYZ colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float], /) -> None:
        """
        Make a XYZ colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.x, self.y, self.z = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.xyz_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return self

    def to_xyy(self) -> xyY:
        return xyY(CC.xyz_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.xyz_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.xyz_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.xyz_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.xyz_to_lch_uv(*self))


class xyY(XYZBased):
    """xyY colourspace object"""
    x: float
    y: float
    Y: float

    peaks: Tuple[float, float] = (0, 1.)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> xyY:
        """
        Make a xyY colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> xyY:
        """
        Make a xyY colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> xyY:
        return _x.to_xyy() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a xyY colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a xyY colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.x, self.y, self.z = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.xyy_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.xyy_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return self

    def to_lab(self) -> Lab:
        return Lab(CC.xyy_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.xyy_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.xyy_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.xyy_to_lch_uv(*self))


class Lab(XYZBased):
    """Lab colourspace object based on Cartesian coordinates"""
    L: float
    """Lightness value"""
    a: float
    """
    Relative to the green–red opponent colors,
    with negative values toward green and positive values toward red
    """
    b: float
    """
    The b* axis represents the blue–yellow opponents,
    with negative numbers toward blue and positive toward yellow
    """

    peaks: Tuple[float, float] = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> Lab:
        """
        Make a Lab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> Lab:
        """
        Make a Lab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> Lab:
        return _x.to_lab() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a Lab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a Lab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.a, self.b = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.lab_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lab_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lab_to_xyy(*self))

    def to_lab(self) -> Lab:
        return self

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.lab_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.lab_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.lab_to_lch_uv(*self))


class LCHab(XYZBased):
    """
    LCHab colourspace object based on polar coordinates
    Cylindrical model of the Lab colourspace
    """

    L: float
    """Lightness value"""

    C: float
    """Chroma, relative saturation"""

    H: float
    """Hue angle, angle of the hue in the CIELAB color wheel"""

    peaks: Tuple[float, float] = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> LCHab:
        """
        Make a LCHab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> LCHab:
        """
        Make a LCHab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> LCHab:
        return _x.to_lch_ab() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a LCHab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a LCHab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.C, self.H = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.lch_ab_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lch_ab_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lch_ab_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.lch_ab_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return self

    def to_luv(self) -> Luv:
        return Luv(CC.lch_ab_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.lch_ab_to_lch_uv(*self))


class Luv(XYZBased):
    """Luv colourspace object based on Cartesian coordinates"""

    L: float
    """Lightness value"""
    u: float
    v: float

    peaks: Tuple[float, float] = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> Luv:
        """
        Make a Luv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> Luv:
        """
        Make a Luv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> Luv:
        return _x.to_luv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a Luv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a Luv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.u, self.v = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.luv_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.luv_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.luv_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.luv_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.luv_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return self

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.luv_to_lch_uv(*self))


class LCHuv(XYZBased):
    """
    LCHab colourspace object based on polar coordinates
    Cylindrical model of the Luv colourspace
    """

    L: float
    """Lightness value"""

    C: float
    """Chroma, relative saturation"""

    H: float
    """Hue angle, angle of the hue in the CIELAB color wheel"""

    peaks: Tuple[float, float] = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> LCHuv:
        """
        Make a LCHuv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tuple[float, float, float]) -> LCHuv:
        """
        Make a LCHuv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> LCHuv:
        return _x.to_lch_uv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a LCHuv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tuple[float, float, float]) -> None:
        """
        Make a LCHuv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[float, float, float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.C, self.H = _x

    def to_rgb(self, rgb_type: Type[_RGB_T], /) -> _RGB_T:
        return RGBS(CC.lch_uv_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lch_uv_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lch_uv_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.lch_uv_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.lch_uv_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.lch_uv_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return self
