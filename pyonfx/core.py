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
"""Main core module"""
from __future__ import annotations

__all__ = [
    'Meta', 'Style',
    'Line', 'Word', 'Syllable', 'Char'
]

import copy
from abc import ABC
from fractions import Fraction
from typing import (Any, Dict, Iterable, List, Literal, MutableSequence,
                    Optional, SupportsIndex, TypeVar, cast, overload)

from .colourspace import ASSColor, Opacity
from .convert import ConvertTime
from .font import Font
from .shape import Pixel, Shape
from .types import Alignment

AssTextT = TypeVar('AssTextT', bound='AssText')


class PList(MutableSequence[AssTextT]):
    """PyonFX mutable sequence."""

    _list: List[AssTextT]

    def __init__(self, iterable: Iterable[AssTextT] | None = None) -> None:
        """
        If no argument is given, the constructor creates a new empty list.

        :param iterable:            Iterable object, defaults to None
        """
        self._list = list(iterable) if iterable else []
        super().__init__()

    @overload
    def __getitem__(self, index: SupportsIndex) -> AssTextT:
        ...

    @overload
    def __getitem__(self, index: slice) -> PList[AssTextT]:
        ...

    def __getitem__(self, index: SupportsIndex | slice) -> AssTextT | PList[AssTextT]:
        if isinstance(index, SupportsIndex):
            return self._list[index]
        else:
            return PList(self._list[index])

    @overload
    def __setitem__(self, index: SupportsIndex, value: AssTextT) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[AssTextT]) -> None:
        ...

    def __setitem__(self, index: SupportsIndex | slice, value: AssTextT | Iterable[AssTextT]) -> None:
        if isinstance(index, SupportsIndex) and not isinstance(value, Iterable):
            self._list[index] = value
        elif isinstance(index, slice) and isinstance(value, Iterable):
            self._list[index] = value
        elif isinstance(index, SupportsIndex) and isinstance(value, Iterable):
            raise TypeError('can only assign a value')
        elif isinstance(index, slice) and not isinstance(value, Iterable):
            raise TypeError('can only assign an iterable')
        else:
            raise NotImplementedError

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        del self._list[index]

    def __len__(self) -> int:
        return len(self._list)

    def insert(self, index: int, value: AssTextT) -> None:
        """
        Insert an AssText value before index

        :param index:               Index number
        :param value:               AssText object
        """
        self._list.insert(index, value)

    @overload
    def strip_empty(self, return_new: Literal[False] = False) -> None:
        """
        Removes objects with empty text or a duration of 0

        :param return_new:          If False, works on the current object, defaults to False
        """
        ...

    @overload
    def strip_empty(self, return_new: Literal[True]) -> PList[AssTextT]:
        """
        Removes objects with empty text or a duration of 0

        :param return_new:          If True, returns a new PList
        """
        ...

    def strip_empty(self, return_new: bool = False) -> None | PList[AssTextT]:

        def _strip_check(a: AssTextT) -> bool:
            return a.text.strip() != '' and a.duration > 0

        if not return_new:
            for a in self:
                if not _strip_check(a):
                    self.remove(a)
            return None
        else:
            return PList(a for a in self if _strip_check(a))

    def adjust_indices(self) -> None:
        if not all(isinstance(text, Line) for text in self):
            raise TypeError

        for line_i, line in enumerate(self):
            line = cast(Line, line)
            line.i = line_i
            char_index = 0
            syl_index = 0
            # word_index = 0

            for word_i, word in enumerate(line.words):
                word.i = word_i

                for syl in word.syls:
                    syl.i = syl_index
                    syl_index += 1

                    syl.word_i = word_i

                    for syl_char_i, char in enumerate(syl.chars):
                        char.i = char_index
                        char_index += 1

                        char.word_i = word_i
                        char.syl_i = syl_index
                        char.syl_char_i = syl_char_i



class DataCore(ABC):
    """Abstract DataCore object"""

    def __str__(self) -> str:
        return self._pretty_print(self)

    def __repr__(self) -> str:
        return repr(self.__dict__)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def _pretty_print(cls, obj: DataCore, indent: int = 0, name: Optional[str] = None) -> str:
        if not name:
            out = " " * indent + f'{obj.__class__.__name__}:\n'
        else:
            out = " " * indent + f'{name}: ({obj.__class__.__name__}):\n'

        # Let's print all this object fields
        indent += 4
        for k, v in obj.__dict__.items():
            if isinstance(v, DataCore):
                # Work recursively to print another object
                out += cls._pretty_print(v, indent, k)
            elif isinstance(v, list):
                v = cast(List[DataCore], v)
                for el in v:
                    # Work recursively to print other objects inside a list
                    out += cls._pretty_print(el, indent, k)
            else:
                # Just print a field of this object
                out += " " * indent + f"{k}: {str(v)}\n"

        return out


class Meta(DataCore):
    """
    Meta object contains informations about the Ass.

    More info about each of them can be found on http://docs.aegisub.org/manual/Styles
    """
    wrap_style: int
    """Determines how line breaking is applied to the subtitle line"""
    scaled_border_and_shadow: bool
    """Determines if it has to be used script resolution (*True*) or video resolution (*False*) to scale border and shadow"""
    play_res_x: int
    """Video width"""
    play_res_y: int
    """Video height"""
    audio: str
    """Loaded audio path (absolute)"""
    video: str
    """Loaded video path (absolute)"""
    fps: Fraction
    """FrameRate per Second"""


class Style(DataCore):
    """
    Style object contains a set of typographic formatting rules that is applied to dialogue lines.

    More info about styles can be found on http://docs.aegisub.org/3.2/ASS_Tags/.
    """
    name: str
    """Style name"""
    fontname: str
    """Font name"""
    fontsize: float
    """Font size in points"""
    color1: ASSColor
    """Primary color (fill)"""
    alpha1: Opacity
    """Transparency of color1"""
    color2: ASSColor
    """Secondary color (secondary fill, for karaoke effect)"""
    alpha2: Opacity
    """Transparency of color2"""
    color3: ASSColor
    """Outline (border) color"""
    alpha3: Opacity
    """Transparency color3"""
    color4: ASSColor
    """Shadow color"""
    alpha4: Opacity
    """Transparency of color4"""
    bold: bool
    """Font with bold"""
    italic: bool
    """Font with italic"""
    underline: bool
    """Font with underline"""
    strikeout: bool
    """Font with strikeout"""
    scale_x: float
    """Text stretching in the horizontal direction"""
    scale_y: float
    """Text stretching in the vertical direction"""
    spacing: float
    """Horizontal spacing between letters"""
    angle: float
    """Rotation of the text"""
    border_style: bool
    """*True* for opaque box, *False* for standard outline"""
    outline: float
    """Border thickness value"""
    shadow: float
    """How far downwards and to the right a shadow is drawn"""
    _alignment: int
    margin_l: int
    """Distance from the left of the video frame"""
    margin_r: int
    """Distance from the right of the video frame"""
    margin_v: int
    """Distance from the bottom (or top if alignment >= 7) of the video frame"""
    encoding: int
    """Codepage used to map codepoints to glyphs"""

    @property
    def alignment(self) -> int:
        """
        Alignment of the text

        setter: Set the alignment. Must be in the range 1 <= an <= 9
        """
        return self._alignment

    @alignment.setter
    def alignment(self, an: Alignment) -> None:
        self._alignment = an

    def an_is_left(self) -> bool:
        return self.alignment in {1, 4, 7}

    def an_is_center(self) -> bool:
        return self.alignment in {2, 5, 8}

    def an_is_right(self) -> bool:
        return self.alignment in {3, 6, 9}

    def an_is_top(self) -> bool:
        return self.alignment in {7, 8, 9}

    def an_is_middle(self) -> bool:
        return self.alignment in {4, 5, 6}

    def an_is_bottom(self) -> bool:
        return self.alignment in {1, 2, 3}


class PositionedText(DataCore, ABC):
    _rounding: int

    x: float
    """Text position horizontal (depends on alignment)"""
    y: float
    """Text position vertical (depends on alignment)."""
    left: float
    """Text position left"""
    center: float
    """Text position center"""
    right: float
    """Text position right"""
    top: float
    """Text position top"""
    middle: float
    """Text position middle"""
    bottom: float
    """Text position bottom"""

    def __init__(self) -> None:
        self._rounding = 3
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'x', 'y', 'left', 'center', 'right', 'top', 'middle', 'bottom'}:
            value = round(value, self._rounding)
        return super().__setattr__(name, value)


class AssText(PositionedText, ABC):
    """Abstract AssText object"""
    i: int
    """Index number"""
    start_time: float
    """Start time (in seconds)"""
    end_time: float
    """End time (in seconds)"""
    duration: float
    """Duration (in seconds)"""
    text: str
    """Text"""
    style: Style
    """Reference to the Style object"""
    meta: Meta
    """Reference to the Meta object"""
    width: float
    """Text width"""
    height: float
    """Text height"""
    ascent: float
    """Font ascent"""
    descent: float
    """Font descent"""
    internal_leading: float
    """Internal leading"""
    external_leading: float
    """External leading"""

    def __copy__(self: AssTextT) -> AssTextT:
        return self

    def __deepcopy__(self: AssTextT, *args: Any) -> AssTextT:
        return self

    def deep_copy(self: AssTextT) -> AssTextT:
        """
        :return:            A deep copy of this object
        """
        return copy.deepcopy(self)

    def shallow_copy(self: AssTextT) -> AssTextT:
        """
        :return:            A shallow copy of this object
        """
        return copy.copy(self)

    def to_shape(self, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
        """
        Converts text with given style information to an ASS shape.
        **Tips:** *You can easily create impressive deforming effects.*

        Examples:
            ..  code-block:: python

                l = line.deep_copy()
                l.text = f"{\\\\an7\\\\pos({line.left},{line.top})\\\\p1}{line.to_shape()}"
                io.write_line(l)

        :param fscx:        The scale_x value for the shape, defaults to None
        :param fscy:        The scale_y value for the shape, defaults to None
        :return:            A Shape object, representing the text with the style format values
                            of the object
        """
        # Obtaining information and editing values of style if requested
        obj = self.deep_copy()

        # Editing temporary the style to properly get the shape
        if fscx is not None:
            obj.style.scale_x = fscx
        if fscy is not None:
            obj.style.scale_y = fscy

        # Obtaining font information from style and obtaining shape
        font = Font(obj.style)
        shape = font.text_to_shape(obj.text)
        # Clearing resources to not let overflow errors take over
        del font, obj

        return shape

    def to_clip(self, an: Alignment, fscx: Optional[float] = None, fscy: Optional[float] = None) -> Shape:
        """
        Converts text with given style information to an ASS shape, applying some translation/scaling to it
        since it is not possible to position a shape with \\pos() once it is in a clip.
        **Tips:** *You can easily create text masks even for growing/shrinking text without too much effort.*

        Examples:
            ..  code-block:: python

                l = line.deep_copy()
                l.text = f"{\\\\an5\\\\pos({line.center},{line.middle})\\\\clip({line.to_clip()})}{line.text}"
                io.write_line(l)

        :param an:          Alignment wanted for the shape
        :param fscx:        The scale_x value for the shape, defaults to None
        :param fscy:        The scale_y value for the shape, defaults to None
        :return:            A Shape object, representing the text with the style format values of the object
        """
        obj = self.deep_copy()

        # Setting default values
        if fscx is None:
            fscx = obj.style.scale_x
        if fscy is None:
            fscy = obj.style.scale_y

        # Obtaining text converted to shape
        shape = obj.to_shape(fscx, fscy)

        # Setting mult_x based on alignment
        if an in {1, 4, 7}:
            mult_x = 0.0
        elif an in {2, 5, 8}:
            mult_x = 1 / 2
        else:
            mult_x = 1.0

        # Setting mult_y based on alignment
        if an in {1, 2, 3}:
            mult_y = 1.0
        elif an in {4, 5, 6}:
            mult_y = 1 / 2
        else:
            mult_y = 0.0

        # Calculating offsets
        cx = obj.left - obj.width * mult_x * (fscx - obj.style.scale_x) / obj.style.scale_x
        cy = obj.top - obj.height * mult_y * (fscy - obj.style.scale_y) / obj.style.scale_y

        shape.move(cx, cy)

        del obj

        return shape

    def to_pixels(self, supersampling: int = 4, anti_aliasing: bool = True) -> List[Pixel]:
        """
        Converts text with given style information to a list of Pixel.
        It is strongly recommended to create a dedicated style for pixels,
        thus, you will write less tags for line in your pixels,
        which means less size for your .ass file.

        Style suggested as an=7, bord=0, shad=0

        :param supersampling:   Supersampling value.
                                Higher value means smoother and more precise anti-aliasing, defaults to 4
        :return:                A list of Pixel representing each individual pixel of the input text styled.
        """
        return self.to_shape().to_pixels(supersampling, anti_aliasing)


class Line(AssText):
    """
    Line object contains informations about a single line in the Ass.

    Note:
        (*) = This field is available only if :class:`extended<Ass>` = True
    """
    comment: bool
    """If *True*, this line will not be displayed on the screen"""
    layer: int
    """Layer for the line. Higher layer numbers are drawn on top of lower ones"""
    leadin: float
    """Time between this line and the previous one (in seconds; first line = 1.001) (*)"""
    leadout: float
    """Time between this line and the next one (in seconds; first line = 1.001) (*)"""
    actor: str
    """Actor field"""
    margin_l: int
    """Left margin for this line"""
    margin_r: int
    """Right margin for this line"""
    margin_v: int
    """Vertical margin for this line"""
    effect: str
    """Effect field"""
    raw_text: str
    """Line raw text"""
    words: PList[Word]
    """List containing objects :class:`Word` in this line (*)"""
    # syls: PList[Syllable]
    # """List containing objects :class:`Syllable` in this line (if available) (*)"""
    # chars: PList[Char]
    # """List containing objects :class:`Char` in this line (*)"""

    @property
    def syls(self) -> PList[Syllable]:
        syls: PList[Syllable] = PList()
        for word in self.words:
            syls.extend(word.syls)
        return syls

    @property
    def chars(self) -> PList[Char]:
        chars: PList[Char] = PList()
        for word in self.words:
            for syl in word.syls:
                chars.extend(syl.chars)
        return chars

    def compose_ass_line(self) -> str:
        """Make an ASS line suitable for writing into ASS file"""
        ass_line = "Comment: " if self.comment else "Dialogue: "
        elements: List[Any] = [
            self.layer,
            ConvertTime.seconds2assts(self.start_time, self.meta.fps),
            ConvertTime.seconds2assts(self.end_time, self.meta.fps),
            self.style.name, self.actor,
            self.margin_l, self.margin_r, self.margin_v,
            self.effect, self.text
        ]
        return ass_line + ','.join(map(str, elements)) + '\n'


class Word(AssText):
    """
    Word object contains informations about a single word of a line in the Ass.

    A word can be defined as some text with some optional space before or after.
    (e.g.: In the string "What a beautiful world!", "beautiful" and "world" are both distinct words).
    """
    prespace: int
    """Word free space before text"""
    postspace: int
    """Word free space after text"""
    syls: PList[Syllable]
    """List containing objects :class:`Syllable` in this line (if available) (*)"""


class WordElement(Word, ABC):
    """Abstract WordElement class"""
    word_i: int
    """Word index (e.g.: In line text ``Hello PyonFX users!``, letter "u" will have word_i=2)"""
    inline_fx: str
    """Inline effect (marked as \\-EFFECT in karaoke-time)"""


class Syllable(WordElement):
    """
    Syllable object contains informations about a single syl of a line in the Ass.

    A syl can be defined as some text after a karaoke tag (k, ko, kf)
    (e.g.: In ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, "Pyon" and "FX" are distinct syllables),
    """
    tags: str
    """All the remaining tags before syl text apart \\k ones"""
    chars: PList[Char]
    """List containing objects :class:`Char` in this line (*)"""


class Char(WordElement):
    """
    Char object contains informations about a single char of a line in the Ass.

    A char is defined by some text between two karaoke tags (k, ko, kf).
    """
    syl_i: int
    """Char syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "F" will have syl_i=3)"""
    syl_char_i: int
    """Char invidual syl index (e.g.: In line text ``{\\k0}Hel{\\k0}lo {\\k0}Pyon{\\k0}FX {\\k0}users!``, letter "e"
    of "users will have syl_char_i=2)"""
