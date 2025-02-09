
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import pytest
import pytest_check as check
from pyonfx import AssUntitled, ConvertTime, Line, logger

folder = Path(__file__).parent
FPS = 24000 / 1001


@pytest.mark.skip
class _Time(NamedTuple):
    start: str
    end: str


@pytest.mark.skip
class _Process:
    io: AssUntitled
    times: list[_Time]

    def __init__(self) -> None:
        self.io = AssUntitled(fps=FPS)

        with open(folder / 'exact_times_per_frame.txt', 'r') as file:
            times = file.readline().split(';')[:100]

        self.times = [_Time(*t.split(',')) for t in times if t]

    @cached_property
    def originals(self) -> list[str]:
        return [
            f'Dialogue: 0,{t.start},{t.end},Default,,0,0,0,,'
            for t in self.times
        ][:20]

    @cached_property
    def processed(self) -> list[Line]:
        return [
            Line.from_text(
                f'Dialogue: 0,{t.start},{t.end},Default,,0,0,0,,',
                i, FPS, self.io.meta, self.io.styles, fix_timestamps=True
            )
            for i, t in enumerate(self.times)
        ][:20]


process = _Process()


def test_convert_time0() -> None:
    for i, (origin, transformed) in enumerate(zip(process.originals, process.processed), start=1):
        _, ostart, oend, *_ = origin.split(',')
        _, tstart, tend, *_ = transformed.as_text(fix_timestamps=True).split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        check.equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


def test_convert_time1() -> None:
    for i, (origin, transformed) in enumerate(zip(process.originals, process.processed), start=1):
        _, ostart, oend, *_ = origin.split(',')
        transformed.start_time += 0.01
        transformed.end_time += 0.01
        _, tstart, tend, *_ = transformed.as_text(fix_timestamps=True).split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        # Exclude broken lines
        if transformed.start_time.ass_frame(FPS, True) != transformed.end_time.ass_frame(FPS, False):
            continue

        check.equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


# def test_convert_time2() -> None:
#     for i, (origin, transformed) in enumerate(zip(process.originals, process.processed), start=1):
#         _, ostart, oend, *_ = origin.split(',')
#         transformed.start_time += 0.02
#         transformed.end_time += 0.02
#         _, tstart, tend, *_ = transformed.as_text(fix_timestamps=True).split(',')
#         logger.trace(ostart + ' | ' + oend)
#         logger.trace(tstart + ' | ' + tend)

#         # Exclude broken lines
#         if transformed.start_time.ass_frame(FPS, True) != transformed.end_time.ass_frame(FPS, False):
#             continue

#         check.equal(
#             ConvertTime.ts2seconds(ostart),
#             ConvertTime.ts2seconds(tstart),
#             msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
#         )
#         check.equal(
#             ConvertTime.ts2seconds(oend),
#             ConvertTime.ts2seconds(tend),
#             msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
#         )


def test_convert_time3() -> None:
    for i, (origin, transformed) in enumerate(zip(process.originals[1:], process.processed), start=1):
        _, ostart, oend, *_ = origin.split(',')
        transformed.start_time += 0.03
        transformed.end_time += 0.03
        _, tstart, tend, *_ = transformed.as_text(fix_timestamps=True).split(',')
        logger.trace(ostart + ' | ' + oend)
        logger.trace(tstart + ' | ' + tend)

        # Exclude broken lines
        if transformed.start_time.ass_frame(FPS, True) != transformed.end_time.ass_frame(FPS, False):
            continue

        check.equal(
            ConvertTime.ts2seconds(ostart),
            ConvertTime.ts2seconds(tstart),
            msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
        )
        check.equal(
            ConvertTime.ts2seconds(oend),
            ConvertTime.ts2seconds(tend),
            msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
        )


# def test_convert_time4() -> None:
#     for i, (origin, transformed) in enumerate(zip(process.originals[1:], process.processed), start=1):
#         _, ostart, oend, *_ = origin.split(',')
#         transformed.start_time += 0.04
#         transformed.end_time += 0.04
#         _, tstart, tend, *_ = transformed.as_text(fix_timestamps=True).split(',')
#         logger.trace(ostart + ' | ' + oend)
#         logger.trace(tstart + ' | ' + tend)

#         # Exclude broken lines
#         if transformed.start_time.ass_frame(FPS, True) != transformed.end_time.ass_frame(FPS, False):
#             continue

#         check.equal(
#             ConvertTime.ts2seconds(ostart),
#             ConvertTime.ts2seconds(tstart),
#             msg='Start line n°' + str(i) + ' | ' + ostart + ' == ' + tstart
#         )
#         check.equal(
#             ConvertTime.ts2seconds(oend),
#             ConvertTime.ts2seconds(tend),
#             msg='End line n°' + str(i) + ' | ' + oend + ' == ' + tend
#         )


if __name__ == '__main__':
    logger.set_level(10)
    # test_convert_time0()
    # test_convert_time1()
    # test_convert_time2()
    # test_convert_time3()
    # test_convert_time4()
