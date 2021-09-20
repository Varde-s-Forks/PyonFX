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
"""Initialisation module"""

from __future__ import annotations

__all__ = ['Ass']

import os
import re
import subprocess
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

from .colourspace import ASSColor, Opacity
from .convert import ConvertTime
from .core import Char, Line, Meta, PList, Style, Syllable, Word
from .font import Font
from .types import AnyPath


class Ass:
    """Initialisation class containing all the information about an ASS file"""
    path_input: Path
    path_output: Path
    fps: Fraction
    vertical_kanji: bool

    meta: Meta
    styles: List[Style]
    lines: List[Line]

    _output: List[str]
    _output_lines: List[str]
    _output_extradata: List[str]

    _ptime: float

    class _TextChunk(NamedTuple):
        tags: str
        text: str
        word_i: Optional[int] = None

    def __init__(self, path_input: AnyPath, path_output: Optional[AnyPath] = None, /,
                 fps: Fraction | float = Fraction(24000, 1001),
                 comment_original: bool = True, extended: bool = True, vertical_kanji: bool = False) -> None:
        """
        :param path_input:          Input file path
        :param path_output:         Output file path, defaults to None
        :param fps:                 Framerate Per Second of the ASS file, defaults to Fraction(24000, 1001)
        :param comment_original:    Keep original lines at the beginning of the output ass and comment them, defaults to True
        :param extended:            Calculate more informations from lines, defaults to True
        :param vertical_kanji:      Line text with alignment 4, 5 or 6 will be positioned vertically
                                    Additionally, ``line`` fields will be re-calculated based on the re-positioned ``line.chars``,
                                    defaults to False
        """
        # Starting to take process time
        self._ptime = time.time()

        # Resolve paths, make absolute the paths

        self.path_input = Path(path_input).resolve()
        if path_output:
            self.path_output = Path(path_output).resolve()
        self.fps = fps if isinstance(fps, Fraction) else Fraction(fps)
        self.vertical_kanji = vertical_kanji
        self._output = []
        self._output_lines = []
        self._output_extradata = []

        self.meta = Meta()
        self.styles = []
        self.lines = []

        # Checking sub file validity (does it exists?)
        if not self.path_input.exists():
            raise FileNotFoundError(f"{path_input} is not a valid path!")

        with open(self.path_input, "r", encoding="utf-8-sig") as file:
            lines_file = file.readlines()

        section = ""
        nbli = 0
        for line_file in lines_file:
            # Getting section
            section_pattern = re.compile(r"^\[([^\]]*)")
            if section_pattern.match(line_file):
                # Updating section
                section_match = section_pattern.match(line_file)
                assert section_match
                section = section_match[1]
                # Appending line to output
                if section != "Aegisub Extradata":
                    self._output.append(line_file)
            elif section in {"Script Info", "Aegisub Project Garbage"}:
                self._parse_meta_data(line_file)
            elif section == "V4+ Styles":
                self._parse_style(line_file)
            elif section == "Events":
                self._parse_dialogue(line_file, nbli, comment_original)
                nbli += 1
            elif section == "Aegisub Extradata":
                self._output_extradata.append(line_file)
            else:
                raise ValueError(f"Unexpected section in the input file: [{section}]")

        # Adding informations to lines and meta?
        if extended:
            lines_by_styles: Dict[Style, List[Line]] = {}
            for line in self.lines:
                # Add dialog text sizes and positions (if possible)
                try:
                    line_style = line.style
                except AttributeError:
                    warnings.warn(f'Line {line.i} is using an undefined style, skipping...', Warning)
                else:
                    # Append dialog to styles (for leadin and leadout later)
                    lines_by_styles.setdefault(line_style, [])
                    lines_by_styles[line_style].append(line)

                    line.duration = line.end_time - line.start_time
                    line.text = re.sub(r"\{.*?\}", "", line.raw_text)

                    font = Font(line_style)
                    font_metrics = font.metrics

                    # Add line data
                    line = self._add_data_line(line, font, font_metrics)

                    # Calculating space width and saving spacing
                    space_width = font.get_text_extents(" ")[0]
                    style_spacing = line_style.spacing

                    # Add words data
                    line = self._add_data_words(line, font, font_metrics, space_width, style_spacing)

                    # Search for dialog's text chunks, to later create syllables
                    # A text chunk is a text with one or more {tags} preceding it
                    # Tags can be some text or empty string
                    text_chunks = self._search_text_chunks(line)

                    # # Adding syls
                    # line = self._add_data_syls(line, font, font_metrics, text_chunks, space_width, style_spacing)

                    # # Getting chars
                    # line = self._add_data_chars(line, font, font_metrics, style_spacing)
                    for word in line.words:
                        word = self._add_data_syls2(word, font, font_metrics, text_chunks, space_width, style_spacing)
                        for syl in word.syls:
                            syl = self._add_data_chars2(syl, font, font_metrics, style_spacing)


            # Add durations between dialogs
            fps = float(self.fps)
            default_lead = 1 / fps * round(fps)
            for _, liness in lines_by_styles.items():
                liness.sort(key=lambda x: x.start_time)
                for li, line in enumerate(liness):
                    line.leadin = (
                        default_lead
                        if li == 0
                        else line.start_time - liness[li - 1].end_time
                    )
                    line.leadout = (
                        default_lead
                        if li == len(liness) - 1
                        else liness[li + 1].start_time - line.end_time
                    )

    def _parse_meta_data(self, line: str) -> None:
        # Switch
        self.meta.fps = self.fps
        if valm := re.match(r"WrapStyle: *?(\d+)$", line):
            self.meta.wrap_style = int(valm[1].strip())
        elif valm := re.match(r"ScaledBorderAndShadow: *?(.+)$", line):
            self.meta.scaled_border_and_shadow = valm[1].strip() == "yes"
        elif valm := re.match(r"PlayResX: *?(\d+)$", line):
            self.meta.play_res_x = int(valm[1].strip())
        elif valm := re.match(r"PlayResY: *?(\d+)$", line):
            self.meta.play_res_y = int(valm[1].strip())
        elif valm := re.match(r"Audio File: *?(.*)$", line):
            if path := self._get_media_abs_path(valm[1].strip()):
                self.meta.audio = path
                line = f"Audio File: {str(self.meta.audio)}\n"
        elif valm := re.match(r"Video File: *?(.*)$", line):
            if path := self._get_media_abs_path(valm[1].strip()):
                self.meta.video = path
                line = f"Video File: {str(self.meta.video)}\n"
        # Appending line to output
        self._output.append(line)

    def _parse_style(self, line: str) -> None:
        self._output.append(line)
        style_match = re.match(r"Style: (.+?)$", line)

        if style_match:
            # Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
            # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,
            # BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
            style = style_match[1].split(",")
            nstyle = Style()

            nstyle.name = str(style[0])

            nstyle.fontname = str(style[1])
            nstyle.fontsize = float(style[2])

            nstyle.color1 = ASSColor(f"&H{style[3][4:]}&")
            nstyle.color2 = ASSColor(f"&H{style[4][4:]}&")
            nstyle.color3 = ASSColor(f"&H{style[5][4:]}&")
            nstyle.color4 = ASSColor(f"&H{style[6][4:]}&")

            nstyle.alpha1 = Opacity.from_ass_val(f"{style[3][:4]}&")
            nstyle.alpha2 = Opacity.from_ass_val(f"{style[4][:4]}&")
            nstyle.alpha3 = Opacity.from_ass_val(f"{style[5][:4]}&")
            nstyle.alpha4 = Opacity.from_ass_val(f"{style[6][:4]}&")

            nstyle.bold = style[7] == "-1"
            nstyle.italic = style[8] == "-1"
            nstyle.underline = style[9] == "-1"
            nstyle.strikeout = style[10] == "-1"

            nstyle.scale_x = float(style[11])
            nstyle.scale_y = float(style[12])

            nstyle.spacing = float(style[13])
            nstyle.angle = float(style[14])

            nstyle.border_style = style[15] == "3"
            nstyle.outline = float(style[16])
            nstyle.shadow = float(style[17])

            nstyle.alignment = int(style[18])
            nstyle.margin_l = int(style[19])
            nstyle.margin_r = int(style[20])
            nstyle.margin_v = int(style[21])

            nstyle.encoding = int(style[22])

            self.styles.append(nstyle)

    def _parse_dialogue(self, line: str, nb: int, comment_original: bool) -> None:
        # Appending line to output (commented) if comment_original is True
        if comment_original:
            self._output.append(
                re.sub(r"^(Dialogue|Comment):", "Comment:", line, count=1)
            )
        elif line.startswith("Format"):
            self._output.append(line.strip())

        # Analysing line
        if anal_line := re.match(r"(Dialogue|Comment): (.+?)$", line):
            # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            nline = Line()

            nline.i = nb

            nline.comment = anal_line[1] == "Comment"
            linesplit = anal_line[2].split(",")

            nline.layer = int(linesplit[0])

            nline.start_time = ConvertTime.assts2seconds(linesplit[1], self.fps)
            nline.start_time = ConvertTime.bound_to_frame(nline.start_time, self.fps)
            nline.end_time = ConvertTime.assts2seconds(linesplit[2], self.fps)
            nline.end_time = ConvertTime.bound_to_frame(nline.end_time, self.fps)

            for style in self.styles:
                if style.name == linesplit[3]:
                    nline.style = style
                    break
            nline.meta = self.meta
            nline.actor = linesplit[4]

            nline.margin_l = int(linesplit[5])
            nline.margin_r = int(linesplit[6])
            nline.margin_v = int(linesplit[7])

            nline.effect = linesplit[8]

            nline.raw_text = ",".join(linesplit[9:])

            self.lines.append(nline)

    def _get_media_abs_path(self, mediafile: str) -> str:
        """
        Internal function that tries to get the absolute path for media files in meta
        If this is not a dummy video, let's try to get the absolute path for the video
        """
        return str(Path(mediafile).resolve()) if not mediafile.startswith("?dummy") else mediafile

    def _add_data_line(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float]) -> Line:
        line.width, line.height = font.get_text_extents(line.text)
        line.ascent, line.descent, line.internal_leading, line.external_leading = font_metrics

        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
        except AttributeError:
            pass
        else:
            # Horizontal position
            margin_l = (
                line.margin_l if line.margin_l != 0 else line.style.margin_l
            )
            margin_r = (
                line.margin_r if line.margin_r != 0 else line.style.margin_r
            )
            if line.style.an_is_left():
                line.left = margin_l
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.left
            elif line.style.an_is_center():
                line.left = play_res_x / 2 - line.width / 2 + margin_l / 2 - margin_r / 2
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.center
            else:
                line.left = play_res_x - margin_r - line.width
                line.center = line.left + line.width / 2
                line.right = line.left + line.width
                line.x = line.right

            # Vertical position
            if line.style.an_is_top():
                line.top = line.margin_v if line.margin_v != 0 else line.style.margin_v
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.top
            elif line.style.an_is_middle():
                line.top = play_res_y / 2 - line.height / 2
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.middle
            else:
                line.top = play_res_y - (line.margin_v if line.margin_v != 0 else line.style.margin_v) - line.height
                line.middle = line.top + line.height / 2
                line.bottom = line.top + line.height
                line.y = line.bottom

        return line

    def _add_data_words(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float],
                        space_width: float, style_spacing: float) -> Line:
        # Adding words
        line.words = PList()

        presp_txt_postsp = re.findall(r"(\s*)([^\s]+)(\s*)", line.text)
        presp_txt_postsp = cast(List[Tuple[str, str, str]], presp_txt_postsp)

        for wi, (prespace, word_text, postspace) in enumerate(presp_txt_postsp):
            word = Word()

            word.i = wi

            word.start_time = line.start_time
            word.end_time = line.end_time
            word.duration = line.duration

            word.style = line.style
            word.text = word_text

            word.prespace = len(prespace)
            word.postspace = len(postspace)

            word.width, word.height = font.get_text_extents(word.text)
            word.ascent, word.descent, word.internal_leading, word.external_leading = font_metrics

            line.words.append(word)

        # Calculate word positions with all words data already available
        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
        except AttributeError:
            pass
        else:
            if line.style.an_is_top() or line.style.an_is_bottom():
                cur_x = line.left
                for word in line.words:
                    # Horizontal position
                    cur_x += word.prespace * space_width + style_spacing
                    word.left = cur_x
                    word.center = word.left + word.width / 2
                    word.right = word.left + word.width

                    if line.style.an_is_left():
                        word.x = word.left
                    elif line.style.an_is_center():
                        word.x = word.center
                    else:
                        word.x = word.right

                    # Vertical position
                    word.top = line.top
                    word.middle = line.middle
                    word.bottom = line.bottom
                    word.y = line.y

                    # Updating cur_x
                    cur_x += word.width + word.postspace * (space_width + style_spacing) + style_spacing
            else:
                max_width, sum_height = 0.0, 0.0
                for word in line.words:
                    max_width = max(max_width, word.width)
                    sum_height = sum_height + word.height

                cur_y = x_fix = play_res_y / 2 - sum_height / 2
                for word in line.words:
                    # Horizontal position
                    x_fix = (max_width - word.width) / 2

                    if line.style.alignment == 4:
                        word.left = line.left + x_fix
                        word.center = word.left + word.width / 2
                        word.right = word.left + word.width
                        word.x = word.left
                    elif line.style.alignment == 5:
                        word.left = play_res_x / 2 - word.width / 2
                        word.center = word.left + word.width / 2
                        word.right = word.left + word.width
                        word.x = word.center
                    else:
                        word.left = line.right - word.width - x_fix
                        word.center = word.left + word.width / 2
                        word.right = word.left + word.width
                        word.x = word.right

                    # Vertical position
                    word.top = cur_y
                    word.middle = word.top + word.height / 2
                    word.bottom = word.top + word.height
                    word.y = word.middle
                    # Updating cur_y
                    cur_y += word.height

        return line

    def _search_text_chunks(self, line: Line) -> List['_TextChunk']:
        # Search for dialog's text chunks, to later create syllables
        # A text chunk is a text with one or more {tags} preceding it
        # Tags can be some text or empty string

        text_chunks = []
        tag_pattern = re.compile(r"(\{.*?\})+")
        tag = tag_pattern.search(line.raw_text)
        word_i = 0

        if not tag:
            # No tags found
            text_chunks.append(self._TextChunk('', line.raw_text))
        else:
            # First chunk without tags?
            if tag.start() != 0:
                text_chunks.append(
                    self._TextChunk('', line.raw_text[0:tag.start()])
                )

            # Searching for other tags
            while True:
                next_tag = tag_pattern.search(line.raw_text, tag.end())
                chk = self._TextChunk(
                    line.raw_text[tag.start() + 1: tag.end() - 1].replace("}{", ""),
                    line.raw_text[tag.end(): (next_tag.start() if next_tag else None)],
                    word_i
                )
                text_chunks.append(chk)

                # If there are some spaces after text, then we're at the end of the current word
                if re.match(r"(.*?)(\s+)$", chk.text):
                    word_i += 1

                if not next_tag:
                    break
                tag = next_tag

        return text_chunks

    def _add_data_syls2(self, word: Word, font: Font, font_metrics: Tuple[float, float, float, float],
                        text_chunks: List['_TextChunk'], space_width: float, style_spacing: float) -> Word:
        # Adding syls
        si = 0
        last_time = 0.0
        inline_fx = ""
        syl_tags_pattern = re.compile(r"(.*?)\\[kK][of]?(\d+)(.*)")

        word.syls = PList()
        for tc in text_chunks:
            # If we don't have at least one \k tag, everything is invalid
            if not syl_tags_pattern.match(tc.tags):
                # word.syls.clear()
                syl = Syllable()
                syl.__dict__ |= word.__dict__
                word.syls.append(syl)
                break

            posttags = tc.tags
            syls_in_text_chunk: List[Syllable] = []
            while True:
                # Are there \k in posttags?
                tags_syl = syl_tags_pattern.match(posttags)

                if not tags_syl:
                    # Append all the temporary syls, except last one
                    for syl in syls_in_text_chunk[:-1]:
                        curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                        if curr_inline_fx:
                            inline_fx = curr_inline_fx[1]
                        syl.inline_fx = inline_fx

                        # Hidden syls are treated like empty syls
                        syl.prespace, syl.text, syl.postspace = 0, "", 0

                        syl.width, syl.height = font.get_text_extents("")
                        syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                        word.syls.append(syl)

                    # Append last syl
                    syl = syls_in_text_chunk[-1]
                    syl.tags += posttags

                    curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                    if curr_inline_fx:
                        inline_fx = curr_inline_fx[1]
                    syl.inline_fx = inline_fx

                    if tc.text.isspace():
                        syl.prespace, syl.text, syl.postspace = 0, tc.text, 0
                    else:
                        if pstxtps := re.match(r"(\s*)(.*?)(\s*)$", tc.text):
                            prespace, syl.text, postspace = pstxtps.groups()
                            syl.prespace, syl.postspace = len(prespace), len(postspace)

                    syl.width, syl.height = font.get_text_extents(syl.text)
                    syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                    word.syls.append(syl)
                    break

                pretags, kdur, posttags = tags_syl.groups()

                # Create a Syllable object
                syl = Syllable()

                syl.start_time = last_time
                # kdur is in centiseconds
                # Converting in seconds...
                syl.end_time = last_time + int(kdur) / 100
                syl.duration = int(kdur) / 100

                syl.style = word.style
                syl.tags = pretags

                syl.i = si
                if tc.word_i is not None:
                    syl.word_i = tc.word_i

                syls_in_text_chunk.append(syl)

                # Update working variable
                si += 1
                last_time = syl.end_time

        # Calculate syllables positions with all syllables data already available
        if word.syls and hasattr(self.meta, 'play_res_x'):
            if word.style.an_is_top() or word.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = word.left
                for syl in word.syls:
                    cur_x += syl.prespace * (space_width + style_spacing)

                    # Horizontal position
                    syl.left = cur_x
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width

                    if word.style.an_is_left():
                        syl.x = syl.left
                    elif word.style.an_is_center():
                        syl.x = syl.center
                    else:
                        syl.x = syl.right

                    cur_x += syl.width + syl.postspace * (space_width + style_spacing) + style_spacing

                    # Vertical position
                    syl.top = word.top
                    syl.middle = word.middle
                    syl.bottom = word.bottom
                    syl.y = word.y

            else:  # Kanji vertical position
                max_width, sum_height = 0.0, 0.0
                for syl in word.syls:
                    max_width = max(max_width, syl.width)
                    sum_height += syl.height

                cur_y = self.meta.play_res_y / 2 - sum_height / 2

                for syl in word.syls:
                    # Horizontal position
                    x_fix = (max_width - syl.width) / 2
                    if word.style.alignment == 4:
                        syl.left = word.left + x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.left
                    elif word.style.alignment == 5:
                        syl.left = word.center - syl.width / 2
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.center
                    else:
                        syl.left = word.right - syl.width - x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.right

                    # Vertical position
                    syl.top = cur_y
                    syl.middle = syl.top + syl.height / 2
                    syl.bottom = syl.top + syl.height
                    syl.y = syl.middle
                    cur_y += syl.height

        return word

    def _add_data_chars2(self, el: Union[Syllable, Word], font: Font, font_metrics: Tuple[float, float, float, float],
                         style_spacing: float) -> Syllable:
        # Adding chars
        el.chars = PList()

        # If we have syls in line, we prefert to work with them to provide more informations
        # words_or_syls: Union[PList[Syllable], PList[Word]] = line.syls if line.syls else line.words

        # Getting chars
        char_index = 0
        el_text = "{}{}{}".format(" " * el.prespace, el.text, " " * el.postspace)
        for ci, char_text in enumerate(el_text):
            char = Char()
            char.i = char_index
            char_index += 1

            # # If we're working with syls, we can add some indexes
            # if line.syls:
            #     # el = cast(Syllable, el)
            #     char.word_i = el.word_i  # type: ignore
            #     char.syl_i = el.i
            #     char.syl_char_i = ci
            # else:
            #     el = cast(Word, el)
            #     char.word_i = el.i
            try:
                char.word_i = el.word_i
            except AttributeError:
                char.word_i = el.i
            char.syl_i = el.i
            char.syl_char_i = ci

            # Adding last fields based on the existance of syls or not
            char.start_time = el.start_time
            char.end_time = el.end_time
            char.duration = el.duration

            char.style = el.style
            char.text = char_text

            char.width, char.height = font.get_text_extents(char.text)
            char.ascent, char.descent, char.internal_leading, char.external_leading = font_metrics

            el.chars.append(char)

        # Calculate character positions with all characters data already available
        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
        except AttributeError:
            pass
        else:
            if el.style.an_is_top() or el.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = el.left
                for char in el.chars:
                    # Horizontal position
                    char.left = cur_x
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width

                    if el.style.an_is_left():
                        char.x = char.left
                    if el.style.an_is_center():
                        char.x = char.center
                    else:
                        char.x = char.right

                    cur_x += char.width + style_spacing

                    # Vertical position
                    char.top = el.top
                    char.middle = el.middle
                    char.bottom = el.bottom
                    char.y = el.y
            else:
                max_width, sum_height = 0.0, 0.0
                for char in el.chars:
                    max_width = max(max_width, char.width)
                    sum_height = sum_height + char.height

                cur_y = x_fix = play_res_y / 2 - sum_height / 2

                # Fixing line positions
                el.top = cur_y
                el.middle = play_res_y / 2
                el.bottom = el.top + sum_height
                el.width = max_width
                el.height = sum_height
                if el.style.alignment == 4:
                    el.center = el.left + max_width / 2
                    el.right = el.left + max_width
                elif el.style.alignment == 5:
                    el.left = el.center - max_width / 2
                    el.right = el.left + max_width
                else:
                    el.left = el.right - max_width
                    el.center = el.left + max_width / 2

                for char in el.chars:
                    # Horizontal position
                    x_fix = (max_width - char.width) / 2
                    if el.style.alignment == 4:
                        char.left = el.left + x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.left
                    elif el.style.alignment == 5:
                        char.left = play_res_x / 2 - char.width / 2
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.center
                    else:
                        char.left = el.right - char.width - x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.right

                    # Vertical position
                    char.top = cur_y
                    char.middle = char.top + char.height / 2
                    char.bottom = char.top + char.height
                    char.y = char.middle
                    cur_y += char.height

        return el


    def _add_data_syls(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float],
                       text_chunks: List['_TextChunk'], space_width: float, style_spacing: float) -> Line:
        # Adding syls
        si = 0
        last_time = 0.0
        inline_fx = ""
        syl_tags_pattern = re.compile(r"(.*?)\\[kK][of]?(\d+)(.*)")

        line.syls = PList()
        for tc in text_chunks:
            # If we don't have at least one \k tag, everything is invalid
            if not syl_tags_pattern.match(tc.tags):
                line.syls.clear()
                break

            posttags = tc.tags
            syls_in_text_chunk: List[Syllable] = []
            while True:
                # Are there \k in posttags?
                tags_syl = syl_tags_pattern.match(posttags)

                if not tags_syl:
                    # Append all the temporary syls, except last one
                    for syl in syls_in_text_chunk[:-1]:
                        curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                        if curr_inline_fx:
                            inline_fx = curr_inline_fx[1]
                        syl.inline_fx = inline_fx

                        # Hidden syls are treated like empty syls
                        syl.prespace, syl.text, syl.postspace = 0, "", 0

                        syl.width, syl.height = font.get_text_extents("")
                        syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                        line.syls.append(syl)

                    # Append last syl
                    syl = syls_in_text_chunk[-1]
                    syl.tags += posttags

                    curr_inline_fx = re.search(r"\\\-([^\\]+)", syl.tags)
                    if curr_inline_fx:
                        inline_fx = curr_inline_fx[1]
                    syl.inline_fx = inline_fx

                    if tc.text.isspace():
                        syl.prespace, syl.text, syl.postspace = 0, tc.text, 0
                    else:
                        if pstxtps := re.match(r"(\s*)(.*?)(\s*)$", tc.text):
                            prespace, syl.text, postspace = pstxtps.groups()
                            syl.prespace, syl.postspace = len(prespace), len(postspace)

                    syl.width, syl.height = font.get_text_extents(syl.text)
                    syl.ascent, syl.descent, syl.internal_leading, syl.external_leading = font_metrics

                    line.syls.append(syl)
                    break

                pretags, kdur, posttags = tags_syl.groups()

                # Create a Syllable object
                syl = Syllable()

                syl.start_time = last_time
                # kdur is in centiseconds
                # Converting in seconds...
                syl.end_time = last_time + int(kdur) / 100
                syl.duration = int(kdur) / 100

                syl.style = line.style
                syl.tags = pretags

                syl.i = si
                if tc.word_i is not None:
                    syl.word_i = tc.word_i

                syls_in_text_chunk.append(syl)

                # Update working variable
                si += 1
                last_time = syl.end_time

        # Calculate syllables positions with all syllables data already available
        if line.syls and hasattr(self.meta, 'play_res_x'):
            if line.style.an_is_top() or line.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = line.left
                for syl in line.syls:
                    cur_x += syl.prespace * (space_width + style_spacing)

                    # Horizontal position
                    syl.left = cur_x
                    syl.center = syl.left + syl.width / 2
                    syl.right = syl.left + syl.width

                    if line.style.an_is_left():
                        syl.x = syl.left
                    elif line.style.an_is_center():
                        syl.x = syl.center
                    else:
                        syl.x = syl.right

                    cur_x += syl.width + syl.postspace * (space_width + style_spacing) + style_spacing

                    # Vertical position
                    syl.top = line.top
                    syl.middle = line.middle
                    syl.bottom = line.bottom
                    syl.y = line.y

            else:  # Kanji vertical position
                max_width, sum_height = 0.0, 0.0
                for syl in line.syls:
                    max_width = max(max_width, syl.width)
                    sum_height += syl.height

                cur_y = self.meta.play_res_y / 2 - sum_height / 2

                for syl in line.syls:
                    # Horizontal position
                    x_fix = (max_width - syl.width) / 2
                    if line.style.alignment == 4:
                        syl.left = line.left + x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.left
                    elif line.style.alignment == 5:
                        syl.left = line.center - syl.width / 2
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.center
                    else:
                        syl.left = line.right - syl.width - x_fix
                        syl.center = syl.left + syl.width / 2
                        syl.right = syl.left + syl.width
                        syl.x = syl.right

                    # Vertical position
                    syl.top = cur_y
                    syl.middle = syl.top + syl.height / 2
                    syl.bottom = syl.top + syl.height
                    syl.y = syl.middle
                    cur_y += syl.height

        return line

    def _add_data_chars(self, line: Line, font: Font, font_metrics: Tuple[float, float, float, float], style_spacing: float) -> Line:
        # Adding chars
        line.chars = PList()

        # If we have syls in line, we prefert to work with them to provide more informations
        words_or_syls: Union[PList[Syllable], PList[Word]] = line.syls if line.syls else line.words

        # Getting chars
        char_index = 0
        for el in words_or_syls:
            el_text = "{}{}{}".format(" " * el.prespace, el.text, " " * el.postspace)
            for ci, char_text in enumerate(el_text):
                char = Char()
                char.i = char_index
                char_index += 1

                # If we're working with syls, we can add some indexes
                if line.syls:
                    # el = cast(Syllable, el)
                    char.word_i = el.word_i  # type: ignore
                    char.syl_i = el.i
                    char.syl_char_i = ci
                else:
                    # el = cast(Word, el)
                    char.word_i = el.i

                # Adding last fields based on the existance of syls or not
                char.start_time = el.start_time
                char.end_time = el.end_time
                char.duration = el.duration

                char.style = line.style
                char.text = char_text

                char.width, char.height = font.get_text_extents(char.text)
                char.ascent, char.descent, char.internal_leading, char.external_leading = font_metrics

                line.chars.append(char)

        # Calculate character positions with all characters data already available
        try:
            play_res_x = self.meta.play_res_x
            play_res_y = self.meta.play_res_y
        except AttributeError:
            pass
        else:
            if line.style.an_is_top() or line.style.an_is_bottom() or not self.vertical_kanji:
                cur_x = line.left
                for char in line.chars:
                    # Horizontal position
                    char.left = cur_x
                    char.center = char.left + char.width / 2
                    char.right = char.left + char.width

                    if line.style.an_is_left():
                        char.x = char.left
                    if line.style.an_is_center():
                        char.x = char.center
                    else:
                        char.x = char.right

                    cur_x += char.width + style_spacing

                    # Vertical position
                    char.top = line.top
                    char.middle = line.middle
                    char.bottom = line.bottom
                    char.y = line.y
            else:
                max_width, sum_height = 0.0, 0.0
                for char in line.chars:
                    max_width = max(max_width, char.width)
                    sum_height = sum_height + char.height

                cur_y = x_fix = play_res_y / 2 - sum_height / 2

                # Fixing line positions
                line.top = cur_y
                line.middle = play_res_y / 2
                line.bottom = line.top + sum_height
                line.width = max_width
                line.height = sum_height
                if line.style.alignment == 4:
                    line.center = line.left + max_width / 2
                    line.right = line.left + max_width
                elif line.style.alignment == 5:
                    line.left = line.center - max_width / 2
                    line.right = line.left + max_width
                else:
                    line.left = line.right - max_width
                    line.center = line.left + max_width / 2

                for char in line.chars:
                    # Horizontal position
                    x_fix = (max_width - char.width) / 2
                    if line.style.alignment == 4:
                        char.left = line.left + x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.left
                    elif line.style.alignment == 5:
                        char.left = play_res_x / 2 - char.width / 2
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.center
                    else:
                        char.left = line.right - char.width - x_fix
                        char.center = char.left + char.width / 2
                        char.right = char.left + char.width
                        char.x = char.right

                    # Vertical position
                    char.top = cur_y
                    char.middle = char.top + char.height / 2
                    char.bottom = char.top + char.height
                    char.y = char.middle
                    cur_y += char.height

        return line

    def get_data(self) -> Tuple[Meta, List[Style], List[Line]]:
        """
        Utility function to easily retrieve meta, styles and lines.

        :return:            :attr:`meta`, :attr:`styles` and :attr:`lines`
        """
        return self.meta, self.styles, self.lines

    def write_line(self, line: Line) -> None:
        """
        Appends a line to the output list that later on will be written to the output file
        when calling save().
        Use it whenever you've prepared a line, it will not impact performance
        since you will not actually write anything until :func:`save` will be called.

        :param line:        Line object
        """
        self._output_lines += [line.compose_ass_line()]

    def save(self, quiet: bool = False) -> None:
        """
        Write everything inside the output list to a file.

        :param quiet:       Don't show message, defaults to False
        """
        with self.path_output.open("w", encoding="utf-8-sig") as file:
            file.writelines(self._output + self._output_lines + ["\n"])
            if self._output_extradata:
                file.write("\n[Aegisub Extradata]\n")
                file.writelines(self._output_extradata)

        if not quiet:
            print(
                f"Produced lines: {len(self._output_lines)}\n"
                f"Process duration (in seconds): {round(time.time() - self._ptime, ndigits=3)}"
            )

    def open_aegisub(self) -> None:
        """
        Open the output (specified in path_output during the initialisation class) with Aegisub.
        """
        # Check if it was saved
        if not self.path_output.exists():
            warnings.warn(
                f'{self.__class__.__name__}: "path_output" not found!', Warning
            )
        else:
            if sys.platform == "win32":
                os.startfile(self.path_output)
            else:
                try:
                    subprocess.call(["aegisub", self.path_output])
                except FileNotFoundError:
                    warnings.warn("Aegisub not found!", Warning)

    def open_mpv(self, video_path: os.PathLike[str] | str | None = None,
                 video_start: Optional[str] = None, full_screen: bool = False) -> None:
        """
        Open the output (specified in path_output during the initialisation class)
        in softsub with MPV player.
        You should add MPV in your PATH (check https://pyonfx.readthedocs.io/en/latest/quick%20start.html#installation-extra-step).

        :param video_path:          Video path. If not specified it will use the path in meta.video, defaults to None
        :param video_start:         Start time for the video (more info: https://mpv.io/manual/master/#options-start)
                                    If not specified, 0 is automatically taken, defaults to None
        :param full_screen:         Launch MPV in full screen, defaults to False
        """
        # Check if it was saved
        if not self.path_output.exists():
            warnings.warn(
                f'{self.__class__.__name__}: "path_output" not found!', Warning
            )
        else:
            # Check if mpv is usable
            if self.meta.video.startswith("?dummy") and not video_path:
                warnings.warn(
                    'Cannot use MPV (if you have it in your PATH) for file preview, since your .ass contains a dummy video.\n'
                    'You can specify a new video source using video_path parameter, check the documentation of the function.',
                    Warning
                )
            else:
                # Setting up the command to execute
                cmd = ["mpv"]

                if video_path:
                    cmd.append(str(video_path))
                else:
                    cmd.append(self.meta.video)
                if video_start:
                    cmd.append("--start=" + video_start)
                if full_screen:
                    cmd.append("--fs")

                cmd.append("--sub-file=" + str(self.path_output))

                try:
                    subprocess.call(cmd)
                except FileNotFoundError:
                    warnings.warn(
                        "MPV not found in your environment variables.\n"
                        "Please refer to the documentation's \"Quick Start\" section if you don't know how to solve it.",
                        Warning
                    )
