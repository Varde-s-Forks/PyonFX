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

from __future__ import annotations

__all__ = ['FrameUtility', 'ColorUtility', 'interpolate']

import re
from typing import TYPE_CHECKING, Any, Dict, Final, Iterable, Iterator, List, NamedTuple, Optional, cast, overload

from typing_extensions import TypeGuard

from .colourspace import ColourSpace
from .geometry import Geometry, Point, PointCartesian3D
from .types import Nb, TCV_co

if TYPE_CHECKING:
    from .core import Line


@overload
def interpolate(val1: Nb, val2: Nb, pct: float = 0.5, acc: float = 1.0) -> Nb:
    ...


@overload
def interpolate(val1: ColourSpace[TCV_co], val2: ColourSpace[TCV_co], pct: float = 0.5, acc: float = 1.0) -> ColourSpace[TCV_co]:
    ...


@overload
def interpolate(val1: List[Point], val2: None = ..., pct: float = 0.5, acc: float = 1.0) -> PointCartesian3D:
    ...


def interpolate(val1: object, val2: Optional[object] = None, pct: float = 0.5, acc: float = 1.0) -> Any:
    """
    Interpolate val1 and val2 (ColourSpace objects or numbers) by percent value

    :param val1:        First value to interpolate
    :param val2:        Second value to interpolate
    :param pct:         Percent value of the interpolation
    :param acc:         Optional acceleration, defaults to 1.0
    :return:            Interpolated value of val1 and val2
    """
    pct = pct ** acc

    if isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
        return val1 * (1 - pct) + val2 * pct
    if isinstance(val1, ColourSpace) and isinstance(val2, ColourSpace):
        return val1.interpolate(val2, pct)

    val1 = cast(List[object], val1)
    if _is_point_seq(val1):
        return Geometry.point_on_bézier_curve(val1, pct)

    raise ValueError(f'interpolate: couldn\'t interpolate val1 "{val1}" and val2 "{val2}"')


def _is_point_seq(val: List[object]) -> TypeGuard[List[Point]]:
    return all(isinstance(p, Point) for p in val)


class Utils:
    """
    This class is a collection of static methods that will help the user in some tasks.
    """

    # @staticmethod
    # def accelerate(pct: float, accelerator: float) -> float:
    #     # Modifies pct according to the acceleration provided.
    #     # TODO: Implement acceleration based on bezier's curve
    #     return pct ** accelerator



NTSC_24P_MS_FROM_FRAME: Final[float] = 1 / (24000 / 1001)


class Frame(NamedTuple):
    """Simple NamedTuple depicting a frame"""
    start: float
    """Start time in seconds"""
    end: float
    """End time in seconds"""
    i: int
    """Index number"""
    n: int
    """Total number of frames"""


class FrameUtility(Iterable[Frame]):
    """Helper class for frame-per-frame calculation"""
    start_time: float
    end_time: float
    frame_dur: float
    n: int

    current_time: float

    def __init__(self, start_time: float, end_time: float, frame_dur: float = NTSC_24P_MS_FROM_FRAME) -> None:
        """
        Examples:
            ..  code-block:: python

                for s, e, i, n in FrameUtility(0, 0.250):
                    print(f"Frame {i}/{n}: {round(s, 3)} - {round(e, 3)}")

                for frame in FrameUtility(0, 0.250):
                    print(
                        f"Frame {frame.i}/{frame.n}: "
                        f'{round(frame.start, 3)} - {round(frame.end, 3)}'
                    )

            >>> Frame 0/6: 0.0 - 0.042
            >>> Frame 1/6: 0.042 - 0.083
            >>> Frame 2/6: 0.083 - 0.125
            >>> Frame 3/6: 0.125 - 0.167
            >>> Frame 4/6: 0.167 - 0.209
            >>> Frame 5/6: 0.209 - 0.25

        :param start_time:      Start time in seconds
        :param end_time:        End time in seconds
        :param frame_dur:       Frame duration, defaults to NTSC_24P_MS_FROM_FRAME
        """
        if end_time < start_time:
            raise ValueError(f"{self.__class__.__name__}: start_time must be > to end_time")

        self.n = round((end_time - start_time) / frame_dur)
        self.start_time = start_time
        self.end_time = end_time
        self.frame_dur = frame_dur
        self.current_time = start_time

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n):
            s = self.start_time + self.frame_dur * i
            e = self.start_time + self.frame_dur * (i + 1) if i < self.n - 1 else self.end_time
            self.current_time = s
            yield Frame(s, e, i, self.n)

        self.current_time = self.start_time


    def add(self, start_time: float, end_time: float, end_value: float, acc: float = 1.0) -> float:
        """
        This function makes a lot easier the calculation of tags value.
        You can see this as a \"\\t\" tag usable in frame per frame operations.
        Use it in a for loop which iterates a FrameUtility object, as you can see in the example.

        Examples:
            ..  code-block:: python

                for frame in (fu := FrameUtility(0, 0.230)):
                    fsc = 100
                    fsc += fu.add(0., 0.075, 50)
                    fsc += fu.add(0.075, 0.175, -50)
                    print(
                        f"Frame {frame.index}/{frame.total}: "
                        f'{round(frame.start, 3)} - {round(frame.end, 3)}'
                        f' | fsc: {round(fsc, 3)}'
                    )

            >>> Frame 0/6: 0.0 - 0.042 | fsc: 100.0
            >>> Frame 1/6: 0.042 - 0.083 | fsc: 127.806
            >>> Frame 2/6: 0.083 - 0.125 | fsc: 145.792
            >>> Frame 3/6: 0.125 - 0.167 | fsc: 124.938
            >>> Frame 4/6: 0.167 - 0.209 | fsc: 104.083
            >>> Frame 5/6: 0.209 - 0.23 | fsc: 100

        :param start_time:      Start time
        :param end_time:        End time
        :param end_value:       Value reached at end_time
        :param acc:             Acceleration value in interpolate function, defaults to 1.0
        :return:                Interpolated value
        """
        if self.current_time < start_time:
            return 0.
        elif self.current_time > end_time:
            return end_value

        pstart = self.current_time - self.start_time - start_time
        pend = end_time - start_time
        return interpolate(0, end_value, pstart / pend, acc)


class ColorUtility:
    """
    This class helps to obtain all the color transformations written in a list of lines
    (usually all the lines of your input .ass)
    to later retrieve all of those transformations that fit between the start_time and end_time of a line passed,
    without having to worry about interpolating times or other stressfull tasks.

    It is highly suggested to create this object just one time in your script, for performance reasons.

    Note:
        A few notes about the color transformations in your lines:

        * Every color-tag has to be in the format of ``c&Hxxxxxx&``, do not forget the last &;
        * You can put color changes without using transformations, like
        ``{\\1c&HFFFFFF&\\3c&H000000&}Test``, but those will be interpreted as ``{\\t(0,0,\\1c&HFFFFFF&\\3c&H000000&)}Test``;
        * For an example of how color changes should be put in your lines, check
        `this <https://github.com/CoffeeStraw/PyonFX/blob/master/examples/2%20-%20Beginner/in2.ass#L34-L36>`_.

        Also, it is important to remember that **color changes in your lines are treated as if they were continuous**.

        For example, let's assume we have two lines:

        #. ``{\\1c&HFFFFFF&\\t(100,150,\\1c&H000000&)}Line1``, starting at 0ms, ending at 100ms;
        #. ``{}Line2``, starting at 100ms, ending at 200ms.

        Even if the second line **doesn't have any color changes** and you would expect to have the style's colors,
        **it will be treated as it has** ``\\1c&H000000&``. That could seem strange at first,
        but thinking about your generated lines, **the majority** will have **start_time and end_time different**
        from the ones of your original file.

        Treating transformations as if they were continous, **ColorUtility will always know the right colors** to pick for you.
        Also, remember that even if you can't always see them directly on Aegisub, you can use transformations
        with negative times or with times that exceed line total duration.

    Parameters:
        lines (list of Line): List of lines to be parsed
        offset (integer, optional): Milliseconds you may want to shift all the color changes

    Returns:
        Returns a ColorUtility object.

    Examples:
        ..  code-block:: python3
            :emphasize-lines: 2, 4

            # Parsing all the lines in the file
            CU = ColorUtility(lines)
            # Parsing just a single line (the first in this case) in the file
            CU = ColorUtility([ line[0] ])
    """

    color_changes: List[Dict[str, Any]]
    c1_req: bool
    c3_req: bool
    c4_req: bool

    def __init__(self, lines: List[Line], offset: int = 0) -> None:
        self.color_changes = []
        self.c1_req = False
        self.c3_req = False
        self.c4_req = False

        # Compiling regex
        tag_all = re.compile(r"{.*?}")
        tag_t = re.compile(r"\\t\( *?(-?\d+?) *?, *?(-?\d+?) *?, *(.+?) *?\)")
        tag_c1 = re.compile(r"\\1c(&H.{6}&)")
        tag_c3 = re.compile(r"\\3c(&H.{6}&)")
        tag_c4 = re.compile(r"\\4c(&H.{6}&)")

        for line in lines:
            # Obtaining all tags enclosured in curly brackets
            tags = tag_all.findall(line.raw_text)

            # Let's search all color changes in the tags
            for tag in tags:
                # Get everything beside \t to see if there are some colors there
                other_tags = tag_t.sub("", tag)

                # Searching for colors in the other tags
                c1, c3, c4 = (
                    tag_c1.search(other_tags),
                    tag_c3.search(other_tags),
                    tag_c4.search(other_tags),
                )

                # If we found something, add to the list as a color change
                if c1 or c3 or c4:
                    if c1:
                        c1 = c1.group(0)  # type: ignore[assignment]
                        self.c1_req = True
                    if c3:
                        c3 = c3.group(0)  # type: ignore[assignment]
                        self.c3_req = True
                    if c4:
                        c4 = c4.group(0)  # type: ignore[assignment]
                        self.c4_req = True

                    self.color_changes.append(
                        {
                            "start": line.start_time + offset,
                            "end": line.start_time + offset,
                            "acc": 1,
                            "c1": c1,
                            "c3": c3,
                            "c4": c4,
                        }
                    )

                # Find all transformation in tag
                ts = tag_t.findall(tag)

                # Working with each transformation
                for t in ts:
                    # Parsing start, end, optional acceleration and colors
                    start, end, acc_colors = int(t[0]), int(t[1]), t[2].split(",")
                    acc, c1, c3, c4 = 1., None, None, None

                    # Do we have also acceleration?
                    if len(acc_colors) == 1:
                        c1, c3, c4 = (
                            tag_c1.search(acc_colors[0]),
                            tag_c3.search(acc_colors[0]),
                            tag_c4.search(acc_colors[0]),
                        )
                    elif len(acc_colors) == 2:
                        acc = float(acc_colors[0])
                        c1, c3, c4 = (
                            tag_c1.search(acc_colors[1]),
                            tag_c3.search(acc_colors[1]),
                            tag_c4.search(acc_colors[1]),
                        )
                    else:
                        # This transformation is malformed (too many ','), let's skip this
                        continue

                    # If found, extract from groups
                    if c1:
                        c1 = c1.group(0)  # type: ignore[assignment]
                        self.c1_req = True
                    if c3:
                        c3 = c3.group(0)  # type: ignore[assignment]
                        self.c3_req = True
                    if c4:
                        c4 = c4.group(0)  # type: ignore[assignment]
                        self.c4_req = True

                    # Saving in the list
                    self.color_changes.append(
                        {
                            "start": line.start_time + start + offset,
                            "end": line.start_time + end + offset,
                            "acc": acc,
                            "c1": c1,
                            "c3": c3,
                            "c4": c4,
                        }
                    )

    def get_color_change(self, line: Line, c1: Optional[bool] = None, c3: Optional[bool] = None, c4: Optional[bool] = None) -> str:
        """Returns all the color_changes in the object that fit (in terms of time) between line.start_time and line.end_time.

        Parameters:
            line (Line object): The line of which you want to get the color changes
            c1 (bool, optional): If False, you will not get color values containing primary color
            c3 (bool, optional): If False, you will not get color values containing border color
            c4 (bool, optional): If False, you will not get color values containing shadow color

        Returns:
            A string containing color changes interpolated.

        Note:
            If c1, c3 or c4 is/are None, the script will automatically recognize what you used in the color changes
            in the lines and put only the ones considered essential.

        Examples:
            ..  code-block:: python3
                :emphasize-lines: 6

                # Assume that we have l as a copy of line and we're iterating over all the syl in the current line
                # All the fun stuff of the effect creation...
                l.start_time = line.start_time + syl.start_time
                l.end_time   = line.start_time + syl.end_time

                l.text = "{\\\\an5\\\\pos(%.3f,%.3f)\\\\fscx120\\\\fscy120%s}%s" % (
                    syl.center, syl.middle, CU.get_color_change(l), syl.text
                )
        """
        transform = ""

        # If we don't have user's settings, we set c values
        # to the ones that we previously saved
        if c1 is None:
            c1 = self.c1_req
        if c3 is None:
            c3 = self.c3_req
        if c4 is None:
            c4 = self.c4_req

        # Reading default colors
        base_c1 = "\\1c" + str(line.style.color1)
        base_c3 = "\\3c" + str(line.style.color3)
        base_c4 = "\\4c" + str(line.style.color4)

        for color_change in self.color_changes:
            if color_change["end"] <= line.start_time:
                # Get base colors from this color change, since it is before my current line
                # Last color change written in .ass wins
                if color_change["c1"]:
                    base_c1 = color_change["c1"]
                if color_change["c3"]:
                    base_c3 = color_change["c3"]
                if color_change["c4"]:
                    base_c4 = color_change["c4"]
            elif color_change["start"] <= line.end_time:
                # We have found a valid color change, append it to the transform
                start_time = color_change["start"] - line.start_time
                end_time = color_change["end"] - line.start_time

                # We don't want to have times = 0
                start_time = 1 if start_time == 0 else start_time
                end_time = 1 if end_time == 0 else end_time

                transform += "\\t(%d,%d," % (start_time, end_time)

                if color_change["acc"] != 1:
                    transform += str(color_change["acc"])

                if c1 and color_change["c1"]:
                    transform += color_change["c1"]
                if c3 and color_change["c3"]:
                    transform += color_change["c3"]
                if c4 and color_change["c4"]:
                    transform += color_change["c4"]

                transform += ")"

        # Appending default color found, if requested
        if c4:
            transform = base_c4 + transform
        if c3:
            transform = base_c3 + transform
        if c1:
            transform = base_c1 + transform

        return transform

    def get_fr_color_change(self, line: Line, c1: Optional[bool] = None, c3: Optional[bool] = None, c4: Optional[bool] = None) -> str:
        """Returns the single color(s) in the color_changes that fit the current frame (line.start_time) in your frame loop.

        Note:
            If you get errors, try either modifying your \\\\t values or set your **fr parameter** in FU object to **10**.

        Parameters:
            line (Line object): The line of which you want to get the color changes
            c1 (bool, optional): If False, you will not get color values containing primary color.
            c3 (bool, optional): If False, you will not get color values containing border color.
            c4 (bool, optional): If False, you will not get color values containing shadow color.

        Returns:
            A string containing color changes interpolated.

        Examples:
            ..  code-block:: python3
                :emphasize-lines: 5

                # Assume that we have l as a copy of line and we're iterating over all the syl in the current line
                # and we're iterating over the frames
                l.start_time = s
                l.end_time   = e

                l.text = "{\\\\an5\\\\pos(%.3f,%.3f)\\\\fscx120\\\\fscy120%s}%s" % (
                    syl.center, syl.middle, CU.get_fr_color_change(l), syl.text
                )
        """
        # If we don't have user's settings, we set c values
        # to the ones that we previously saved
        if c1 is None:
            c1 = self.c1_req
        if c3 is None:
            c3 = self.c3_req
        if c4 is None:
            c4 = self.c4_req

        # Reading default colors
        base_c1 = "\\1c" + str(line.style.color1)
        base_c3 = "\\3c" + str(line.style.color3)
        base_c4 = "\\4c" + str(line.style.color4)

        # Searching valid color_change
        current_time = line.start_time
        latest_index = -1

        for i, color_change in enumerate(self.color_changes):
            if current_time >= color_change["start"]:
                latest_index = i

        # If no color change is found, take default from style
        if latest_index == -1:
            colors = ""
            if c1:
                colors += base_c1
            if c3:
                colors += base_c3
            if c4:
                colors += base_c4
            return colors

        # If we have passed the end of the lastest color change available, then take the final values of it
        if current_time >= self.color_changes[latest_index]["end"]:
            colors = ""
            if c1 and self.color_changes[latest_index]["c1"]:
                colors += self.color_changes[latest_index]["c1"]
            if c3 and self.color_changes[latest_index]["c3"]:
                colors += self.color_changes[latest_index]["c3"]
            if c4 and self.color_changes[latest_index]["c4"]:
                colors += self.color_changes[latest_index]["c4"]
            return colors

        # Else, interpolate the latest color change
        start = current_time - self.color_changes[latest_index]["start"]
        end = (
            self.color_changes[latest_index]["end"]
            - self.color_changes[latest_index]["start"]
        )
        pct = start / end

        # If we're in the first color_change, interpolate with base colors
        if latest_index == 0:
            colors = ""
            if c1 and self.color_changes[latest_index]["c1"]:
                colors += "\\1c" + interpolate(
                    pct,
                    base_c1[3:],
                    self.color_changes[latest_index]["c1"][3:],
                    self.color_changes[latest_index]["acc"],
                )
            if c3 and self.color_changes[latest_index]["c3"]:
                colors += "\\3c" + interpolate(
                    pct,
                    base_c3[3:],
                    self.color_changes[latest_index]["c3"][3:],
                    self.color_changes[latest_index]["acc"],
                )
            if c4 and self.color_changes[latest_index]["c4"]:
                colors += "\\4c" + interpolate(
                    pct,
                    base_c4[3:],
                    self.color_changes[latest_index]["c4"][3:],
                    self.color_changes[latest_index]["acc"],
                )
            return colors

        # Else, we interpolate between current color change and previous
        colors = ""
        if c1:
            colors += "\\1c" + interpolate(
                pct,
                self.color_changes[latest_index - 1]["c1"][3:],
                self.color_changes[latest_index]["c1"][3:],
                self.color_changes[latest_index]["acc"],
            )
        if c3:
            colors += "\\3c" + interpolate(
                pct,
                self.color_changes[latest_index - 1]["c3"][3:],
                self.color_changes[latest_index]["c3"][3:],
                self.color_changes[latest_index]["acc"],
            )
        if c4:
            colors += "\\4c" + interpolate(
                pct,
                self.color_changes[latest_index - 1]["c4"][3:],
                self.color_changes[latest_index]["c4"][3:],
                self.color_changes[latest_index]["acc"],
            )
        return colors
