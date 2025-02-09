# PyonFX: An easy way to create KFX (Karaoke Effects) and complex typesetting using the ASS format (Advanced Substation Alpha).
# Copyright (C) 2019 Antonio Strippoli (CoffeeStraw/YellowFlash)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyonFX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.
"""Conversion module"""
from __future__ import annotations

__all__ = [
    'ConvertTime', 'ConvertColour'
]

import colorsys
import math
from typing import Final

import numpy as np

from ._logging import logger
from .misc import clamp_value
from .ptypes import Tup3


class ConvertTime:
    """Time conversion class"""
    # Seconds | Timestamp
    @staticmethod
    def ts2seconds(ts: str, /) -> float:
        h, m, s = map(float, ts.split(':'))
        return h * 3600 + m * 60 + s

    @classmethod
    def seconds2ts(cls, s: float, /, *, precision: int = 3) -> str:
        m = s // 60
        s %= 60
        h = m // 60
        m %= 60
        return cls.composets(h, m, s, precision=precision)

    # Seconds | Frame
    @staticmethod
    def seconds2f(s: float, fps: float, /) -> int:
        return round(s * fps)

    @staticmethod
    def f2seconds(f: int, fps: float, /) -> float:
        if f == 0:
            return 0.0

        t = round(float(10 ** 9 * f * fps ** -1))
        s = t / 10 ** 9
        return s

    # Seconds | ASS frame
    # Heavily inspired from https://github.com/CoffeeStraw/PyonFX/pull/37
    @staticmethod
    def seconds2assf(s: float, fps: float, /, is_start: bool) -> int:
        return math.ceil((s - 0.0005) * fps) - (0 if is_start else 1)

    @classmethod
    def assf2seconds(cls, f: int, fps: float, /, is_start: bool) -> float:
        if is_start and f == 0:
            return 0.0

        curr_ms = cls.f2seconds(f, fps) * 1000
        if is_start:
            prev_ms = cls.f2seconds(f - 1, fps) * 1000
            ms = math.floor(prev_ms + int((curr_ms - prev_ms + 1) / 2) + 1e-6)
        else:
            next_ms = cls.f2seconds(f + 1, fps) * 1000
            ms = math.floor(curr_ms + int((next_ms - curr_ms + 1) / 2) + 1e-6)
        s = ms / 1000
        return s

    # Frame | Timestamp
    @classmethod
    def f2ts(cls, f: int, fps: float, /, *, precision: int = 3) -> str:
        s = cls.f2seconds(f, fps)
        ts = cls.seconds2ts(s, precision=precision)
        return ts

    @classmethod
    def ts2f(cls, ts: str, fps: float, /) -> int:
        s = cls.ts2seconds(ts)
        f = cls.seconds2f(s, fps)
        return f

    # Ass Timestamp | Seconds
    @classmethod
    def seconds2assts(cls, s: float, fps: float, /, is_start: bool) -> str:
        s -= fps ** -1 * 0.5
        s = cls.bound2assframe(s, fps, is_start, shifted=True)
        ts = cls.seconds2ts(min(max(0, s), 35999.999), precision=3)
        return ts[1:-1]

    @classmethod
    def assts2seconds(cls, assts: str, fps: float, /, is_start: bool) -> float:
        s = cls.ts2seconds(assts)
        if s == 0.0:
            return s
        s = cls.bound2assframe(s, fps, is_start, shifted=True)
        s += fps ** -1 * 0.5
        return s

    @staticmethod
    @logger.catch(force_exit=True)
    def composets(h: float, m: float, s: float, /, *, precision: int = 3) -> str:
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

    @classmethod
    def bound2frame(cls, s: float, fps: float, /) -> float:
        return cls.f2seconds(cls.seconds2f(s, fps), fps)

    @classmethod
    def bound2assframe(cls, s: float, fps: float, /, is_start: bool, shifted: bool = False) -> float:
        if s == 0.0:
            return s
        if not shifted:
            s -= fps ** -1 * 0.5
        s = cls.assf2seconds(cls.seconds2assf(s, fps, is_start), fps, is_start)
        if not shifted:
            s += fps ** -1 * 0.5
        return s



class ConvertColour:
    """Colour conversion class"""

    # Reference white
    D65_XYZ_TRISTIMULUS_2: Final[Tup3[float]] = (0.95047, 1.00000, 1.08883)
    D65_XYZ_TRISTIMULUS_10: Final[Tup3[float]] = (0.9481, 1.000, 1.07304)
    κ: Final[float] = 24389 / 27
    ϵ: Final[float] = 216 / 24389


    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
        return colorsys.hls_to_rgb(h, l, s)

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
        return colorsys.hsv_to_rgb(h, s, v)

    # -------------------------------------------------------------------------
    # ---------------------------- RGB Conversions ----------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
        rgb_mat = np.array((r, g, b), np.float64)  # type: ignore[var-annotated]
        conv_mat = np.array(  # type: ignore[var-annotated]
            [(0.4124564, 0.3575761, 0.1804375),
             (0.2126729, 0.7151522, 0.0721750),
             (0.0193339, 0.1191920, 0.9503041)],
            np.float64
        )
        inv_srgb_comp = np.where(
            rgb_mat <= 0.04045,
            rgb_mat / 12.92,
            ((rgb_mat + 0.055) / 1.055) ** 2.4
        )
        return tuple(np.dot(conv_mat, np.array([*inv_srgb_comp])))  # type: ignore

    @classmethod
    def rgb_to_xyy(cls, r: float, g: float, b: float) -> tuple[float, float, float]:
        xyz = cls.rgb_to_xyz(r, g, b)
        return cls.xyz_to_xyy(*xyz)

    @classmethod
    def rgb_to_lab(cls, r: float, g: float, b: float) -> tuple[float, float, float]:
        xyz = cls.rgb_to_xyz(r, g, b)
        return cls.xyz_to_lab(*xyz)

    @classmethod
    def rgb_to_lch_ab(cls, r: float, g: float, b: float) -> tuple[float, float, float]:
        xyz = cls.rgb_to_xyz(r, g, b)
        return cls.xyz_to_lch_ab(*xyz)

    @classmethod
    def rgb_to_luv(cls, r: float, g: float, b: float) -> tuple[float, float, float]:
        xyz = cls.rgb_to_xyz(r, g, b)
        return cls.xyz_to_luv(*xyz)

    @classmethod
    def rgb_to_lch_uv(cls, r: float, g: float, b: float) -> tuple[float, float, float]:
        xyz = cls.rgb_to_xyz(r, g, b)
        return cls.xyz_to_lch_uv(*xyz)

    @staticmethod
    def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, s, l

    @staticmethod
    def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(r, g, b)

    # -------------------------------------------------------------------------
    # ---------------------------- XYZ Conversions ----------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def xyz_to_xyy(cls, x: float, y: float, z: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
        if not x == y == z == 0.0:
            Y = y
            x, y = map(lambda a: a / (x + y + z), (x, y))
        else:
            Y = 0.0
            x, y = map(lambda a: a / sum(cls.D65_XYZ_TRISTIMULUS_10), (x, y))
        return x, y, Y

    @classmethod
    def xyz_to_lab(cls, x: float, y: float, z: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
        xr, yr, zr = [a / b for a, b in zip((x, y, z), cls.D65_XYZ_TRISTIMULUS_10)]
        fx, fy, fz = [a ** (1/3) if a > cls.ϵ else (cls.κ * a + 16) / 116 for a in (xr, yr, zr)]
        return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

    @classmethod
    def xyz_to_lch_ab(cls, x: float, y: float, z: float) -> tuple[float, float, float]:
        lab = cls.xyz_to_lab(x, y, z)
        return cls.lab_to_lch_ab(*lab)

    @classmethod
    def xyz_to_luv(cls, x: float, y: float, z: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Luv.html
        if x == y == z == 0.:
            return 0, 0, 0

        x_ref, y_ref, z_ref = cls.D65_XYZ_TRISTIMULUS_10
        yr = y / y_ref
        up = 4 * x / (x + 15 * y + 3 * z)
        vp = 9 * y / (x + 15 * y + 3 * z)

        up_ref = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
        vp_ref = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

        l = 116 * yr ** (1 / 3) - 16 if yr > cls.ϵ else cls.κ * yr
        u = 13 * l * (up - up_ref)
        v = 13 * l * (vp - vp_ref)

        return l, u, v

    @classmethod
    def xyz_to_lch_uv(cls, x: float, y: float, z: float) -> tuple[float, float, float]:
        luv = cls.xyz_to_luv(x, y, z)
        return cls.luv_to_lch_uv(*luv)

    @staticmethod
    def xyz_to_rgb(x: float, y: float, z: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
        xyz_mat = np.array((x, y, z), np.float64)  # type: ignore[var-annotated]
        conv_mat = np.array(  # type: ignore[var-annotated]
            [(3.2404542, -1.5371385, -0.4985314),
             (-0.9692660, 1.8760108, 0.0415560),
             (0.0556434, -0.2040259, 1.0572252)],
            np.float64
        )
        linear_rgb = np.dot(conv_mat, xyz_mat)
        srgb_comp = np.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * linear_rgb ** (1 / 2.4) - 0.055
        )
        return tuple(map(lambda a: clamp_value(a, 0.0, 1.0), srgb_comp))  # type: ignore

    # -------------------------------------------------------------------------
    # ---------------------------- xyY Conversions ----------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def xyy_to_xyz(x: float, y: float, Y: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
        if Y == 0.0:
            x = y = z = 0.0
        else:
            z = ((1 - x - y) * Y) / y
            x = (x * Y) / y
            y = Y
        return x, y, z

    @classmethod
    def xyy_to_lab(cls, x: float, y: float, Y: float) -> tuple[float, float, float]:
        xyz = cls.xyy_to_xyz(x, y, Y)
        return cls.xyz_to_lab(*xyz)

    @classmethod
    def xyy_to_lch_ab(cls, x: float, y: float, Y: float) -> tuple[float, float, float]:
        lab = cls.xyy_to_lab(x, y, Y)
        return cls.lab_to_lch_ab(*lab)

    @classmethod
    def xyy_to_luv(cls, x: float, y: float, Y: float) -> tuple[float, float, float]:
        xyz = cls.xyy_to_xyz(x, y, Y)
        return cls.xyz_to_luv(*xyz)

    @classmethod
    def xyy_to_lch_uv(cls, x: float, y: float, Y: float) -> tuple[float, float, float]:
        luv = cls.xyy_to_luv(x, y, Y)
        return cls.luv_to_lch_uv(*luv)

    @classmethod
    def xyy_to_rgb(cls, x: float, y: float, Y: float) -> tuple[float, float, float]:
        xyz = cls.xyy_to_xyz(x, y, Y)
        return cls.xyz_to_rgb(*xyz)

    # -------------------------------------------------------------------------
    # ---------------------------- Lab Conversions ----------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def lab_to_xyz(cls, l: float, a: float, b: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
        fy = (l + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        yr = fy ** 3 if l > cls.ϵ * cls.κ else l / cls.κ
        xr, zr = map(lambda n: n ** 3 if n ** 3 > cls.ϵ else (116 * n - 16) / cls.κ, (fx, fz))
        return tuple(n * m for n, m in zip((xr, yr, zr), cls.D65_XYZ_TRISTIMULUS_10))  # type: ignore

    @classmethod
    def lab_to_xyy(cls, l: float, a: float, b: float) -> tuple[float, float, float]:
        xyz = cls.lab_to_xyz(l, a, b)
        return cls.xyz_to_xyy(*xyz)

    @staticmethod
    def lab_to_lch_ab(l: float, a: float, b: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_Lab_to_LCH.html
        return l, math.sqrt(a ** 2 + b ** 2), (math.atan2(b, a) * 180 / math.pi) % 360

    @classmethod
    def lab_to_luv(cls, l: float, a: float, b: float) -> tuple[float, float, float]:
        xyz = cls.lab_to_xyz(l, a, b)
        return cls.xyz_to_luv(*xyz)

    @classmethod
    def lab_to_lch_uv(cls, l: float, a: float, b: float) -> tuple[float, float, float]:
        luv = cls.lab_to_luv(l, a, b)
        return cls.luv_to_lch_uv(*luv)

    @classmethod
    def lab_to_rgb(cls, l: float, a: float, b: float) -> tuple[float, float, float]:
        xyz = cls.lab_to_xyz(l, a, b)
        return cls.xyz_to_rgb(*xyz)

    # -------------------------------------------------------------------------
    # --------------------------- LCHab Conversions ---------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def lch_ab_to_xyz(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        lab = cls.lch_ab_to_lab(l, c, h)
        return cls.lab_to_xyy(*lab)

    @classmethod
    def lch_ab_to_xyy(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_ab_to_xyz(l, c, h)
        return cls.xyz_to_xyy(*xyz)

    @staticmethod
    def lch_ab_to_lab(l: float, c: float, h: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_LCH_to_Lab.html
        hr = math.radians(h)
        return l, math.cos(hr) * c, math.sin(hr) * c

    @classmethod
    def lch_ab_to_luv(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_ab_to_xyz(l, c, h)
        return cls.xyz_to_luv(*xyz)

    @classmethod
    def lch_ab_to_lch_uv(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_ab_to_xyz(l, c, h)
        return cls.xyz_to_lch_uv(*xyz)

    @classmethod
    def lch_ab_to_rgb(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_ab_to_xyz(l, c, h)
        return cls.xyz_to_rgb(*xyz)

    # -------------------------------------------------------------------------
    # ---------------------------- Luv Conversions ----------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def luv_to_xyz(cls, l: float, u: float, v: float) -> tuple[float, float, float]:
        x_ref, y_ref, z_ref = cls.D65_XYZ_TRISTIMULUS_10
        u0 = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
        v0 = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

        y = ((l + 16) / 116) ** 3 if l > cls.κ * cls.ϵ else l / cls.κ

        a = ((52 * l) / (u + 13 * l * u0) - 1) * (1 / 3)
        b = (-5) * y
        c = - 1 / 3
        d = ((39 * l) / (v + 13 * l * v0) - 5) * y

        x = (d - b) / (a - c)
        z = x * a + b

        return x, y, z

    @classmethod
    def luv_to_xyy(cls, l: float, u: float, v: float) -> tuple[float, float, float]:
        xyz = cls.luv_to_xyz(l, u, v)
        return cls.xyz_to_xyy(*xyz)

    @classmethod
    def luv_to_lab(cls, l: float, u: float, v: float) -> tuple[float, float, float]:
        xyz = cls.luv_to_xyz(l, u, v)
        return cls.xyz_to_lab(*xyz)

    @classmethod
    def luv_to_lch_ab(cls, l: float, u: float, v: float) -> tuple[float, float, float]:
        lab = cls.luv_to_lab(l, u, v)
        return cls.lab_to_lch_ab(*lab)

    @staticmethod
    def luv_to_lch_uv(l: float, u: float, v: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_Luv_to_LCH.html
        return l, math.sqrt(u ** 2 + v ** 2), (math.degrees(math.atan2(v, u))) % 360

    @classmethod
    def luv_to_rgb(cls, l: float, u: float, v: float) -> tuple[float, float, float]:
        xyz = cls.luv_to_xyz(l, u, v)
        return cls.xyz_to_rgb(*xyz)

    # -------------------------------------------------------------------------
    # --------------------------- LCHuv Conversions ---------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def lch_uv_to_xyz(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        luv = cls.lch_uv_to_luv(l, c, h)
        return cls.luv_to_xyz(*luv)

    @classmethod
    def lch_uv_to_xyy(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_uv_to_xyz(l, c, h)
        return cls.xyz_to_xyy(*xyz)

    @classmethod
    def lch_uv_to_lab(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_uv_to_xyz(l, c, h)
        return cls.xyz_to_lab(*xyz)

    @classmethod
    def lch_uv_to_lch_ab(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_uv_to_xyz(l, c, h)
        return cls.xyz_to_lch_ab(*xyz)

    @staticmethod
    def lch_uv_to_luv(l: float, c: float, h: float) -> tuple[float, float, float]:
        # http://www.brucelindbloom.com/index.html?Eqn_LCH_to_Luv.html
        hr = h * math.pi/180
        return l, math.cos(hr) * c, math.sin(hr) * c

    @classmethod
    def lch_uv_to_rgb(cls, l: float, c: float, h: float) -> tuple[float, float, float]:
        xyz = cls.lch_uv_to_xyz(l, c, h)
        return cls.xyz_to_rgb(*xyz)
