
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any

import win32con
import win32gui
import win32ui
if TYPE_CHECKING:
    from win32helper.win32typing import PyCFont  # type: ignore
else:
    PyCFont = Any

from .._logging import logger
from ..shape import DrawingCommand, DrawingProp, Shape

if TYPE_CHECKING:
    from ..core import Style
else:
    Style = Any

from ._abstract import _AbstractFont, _Metrics, _TextExtents


class Font(_AbstractFont):
    _metrics: dict[str, float]

    pycfont: PyCFont

    def __init__(self, style: Style) -> None:
        super().__init__(style)
        # Create device context
        self.dc = win32gui.CreateCompatibleDC(None)
        # Set context coordinates mapping mode
        win32gui.SetMapMode(self.dc, win32con.MM_TEXT)
        # Set context backgrounds to transparent
        win32gui.SetBkMode(self.dc, win32con.TRANSPARENT)
        # Create font handle
        font_spec: dict[str, Any] = {
            "height": int(self.style.fontsize * self.upscale),
            "width": 0,
            "escapement": 0,
            "orientation": 0,
            "weight": win32con.FW_BOLD if self.style.bold else win32con.FW_NORMAL,
            "italic": int(bool(self.style.italic)),
            "underline": int(bool(self.style.underline)),
            "strike out": int(bool(self.style.strikeout)),
            "charset": win32con.DEFAULT_CHARSET,
            "out precision": win32con.OUT_TT_PRECIS,
            "clip precision": win32con.CLIP_DEFAULT_PRECIS,
            "quality": win32con.ANTIALIASED_QUALITY,
            "pitch and family": win32con.DEFAULT_PITCH + win32con.FF_DONTCARE,
            "name": self.style.fontname,
        }
        self.pycfont = win32ui.CreateFont(font_spec)
        win32gui.SelectObject(self.dc, self.pycfont.GetSafeHandle())
        # Calculate metrics
        self._metrics = win32gui.GetTextMetrics(self.dc)

    def __del__(self) -> None:
        win32gui.DeleteObject(self.pycfont.GetSafeHandle())
        win32gui.DeleteDC(self.dc)
        self.text_extents.cache_clear()
        self.text_to_shape.cache_clear()

    @cached_property
    def metrics(self) -> _Metrics:
        const = self.downscale * self.yscale
        return _Metrics(
            # 'height': self.metrics['Height'] * const,
            self._metrics["Ascent"] * const,
            self._metrics["Descent"] * const,
            self._metrics["InternalLeading"] * const,
            self._metrics["ExternalLeading"] * const,
        )

    @lru_cache(maxsize=None)
    def text_extents(self, text: str) -> _TextExtents:
        cx, cy = win32gui.GetTextExtentPoint32(self.dc, text)

        return _TextExtents(
            (cx * self.downscale + self.hspace * (len(text) - 1)) * self.xscale,
            cy * self.downscale * self.yscale,
        )

    @lru_cache(maxsize=256)
    @logger.catch
    def text_to_shape(self, text: str) -> Shape:
        if not text:
            raise ValueError(f'{self.__class__.__name__}: Text is empty!')
        # TODO: Calcultating distance between origins of character cells (just in case of spacing)

        # Add path to device context
        win32gui.BeginPath(self.dc)
        win32gui.ExtTextOut(self.dc, 0, 0, 0x0, None, text)
        win32gui.EndPath(self.dc)
        # Getting Path produced by Microsoft API
        points, type_points = win32gui.GetPath(self.dc)

        # Checking for errors
        if len(points) == 0 or len(points) != len(type_points):
            raise RuntimeError(
                f'{self.__class__.__name__}: no points detected or mismatch length between points and type_points'
            )

        # Defining variables
        PT_MOVE, PT_LINE, PT_BÉZIER = win32con.PT_MOVETO, win32con.PT_LINETO, win32con.PT_BEZIERTO
        PT_CLOSE = win32con.PT_CLOSEFIGURE
        PT_LINE_OR_CLOSE, PT_BÉZIER_OR_CLOSE = PT_LINE | PT_CLOSE, PT_BÉZIER | PT_CLOSE

        cmds: list[DrawingCommand] = []
        DC, DP = DrawingCommand, DrawingProp
        m, l, b = DP.MOVE, DP.LINE, DP.BÉZIER

        points_types = iter(zip(points, type_points))
        while 1:
            try:
                (x0, y0), ptype = next(points_types)
            except StopIteration:
                break
            if ptype == PT_MOVE:
                cmds.append(DC(m, (x0, y0)))
            elif ptype in {PT_LINE, PT_LINE_OR_CLOSE}:
                cmds.append(DC(l, (x0, y0)))
            elif ptype in {PT_BÉZIER, PT_BÉZIER_OR_CLOSE}:
                (x1, y1), _ = next(points_types)
                (x2, y2), _ = next(points_types)
                cmds.append(DC(b, (x0, y0), (x1, y1), (x2, y2)))

        # Clear device context path
        win32gui.AbortPath(self.dc)

        shape = Shape(cmds)
        shape.scale(self.downscale * self.xscale, self.downscale * self.yscale)
        return shape
