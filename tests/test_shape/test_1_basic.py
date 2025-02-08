# flake8: noqa
import math

import pytest_check as check
from pyonfx import CartesianAxis
from pyonfx import DrawingCommand as DC
from pyonfx import DrawingProp as DP
from pyonfx import Point
from pyonfx import PointCartesian2D as PC2D
from pyonfx import Shape


def test_init() -> None:
    original = Shape([
        DC(DP.MOVE, PC2D(x=400.0, y=1664.0)),
        DC(DP.LINE, PC2D(x=826.67, y=1453.33)),
        DC(DP.LINE, PC2D(x=2437.33, y=930.67)),
        DC(DP.LINE, PC2D(x=2216.0, y=440.0)),
        DC(DP.LINE, PC2D(x=2850.67, y=1530.67)),
        DC(DP.LINE, PC2D(x=1712.0, y=141.33)),
        DC(DP.CUBIC_BÉZIER_CURVE, PC2D(x=851.33, y=-12.0), PC2D(x=156.67, y=905.33), PC2D(x=136.0, y=381.33)),
        DC(DP.CUBIC_BÉZIER_CURVE, PC2D(x=206.67, y=1994.67), PC2D(x=449.33, y=2018.67), PC2D(x=3672.0, y=1469.33)),
    ])
    dest = 'm 400 1664 l 826.67 1453.33 2437.33 930.67 2216 440 2850.67 1530.67 1712 141.33 b 851.33 -12 156.67 905.33 136 381.33 206.67 1994.67 449.33 2018.67 3672 1469.33 '
    check.equal(str(original), str(dest))


def test_round0() -> None:
    original = Shape.from_ass_string("m 0.5 0.4 l 20.5 0.6 20.7 10.1 0.6 10.4")
    dest = Shape.from_ass_string("m 0 0 l 20 1 21 10 1 10")
    original.round(0)
    check.equal(str(original), str(dest))


def test_map1() -> None:
    original = Shape.from_ass_string('m 757.33 832 b 1216.67 1440.67 311.33 1967.33 760 1880 875.33 2140 2167.33 1332 2360 2042.67 3037.33 2083.33 2786.67 1428.67 3621.33 1632 l 3298.67 901.33 2792 1237.33 2600 944 2378.67 1306.67 2074.67 786.67 1770.67 1296 1637.33 765.33 1421.33 1272')
    dest = Shape.from_ass_string('m 1514.66 1664 b 1216.67 1440.67 207.553 1311.553 380 940 350.132 856 722.443 444 674.286 583.62 759.332 520.832 619.26 317.482 724.266 326.4 l 599.758 163.878 465.333 206.222 400 145.231 339.81 186.667 276.623 104.889 221.334 162 192.627 90.039 157.926 141.333')
    var = 1

    def _func(p: Point) -> Point:
        nonlocal var
        p = p.to_2d()
        p.x *= len(p) / var
        p.y *= len(p) / var
        var += 1
        return p
    original.map(_func)
    original.round()
    check.equal(str(original), str(dest))


# TODO: More map


def test_move1() -> None:
    original = Shape.from_ass_string("m 0 0 l 20 0 20 10 0 10")
    dest = Shape.from_ass_string("m 10 5 l 30 5 30 15 10 15")
    original.move(10, 5)
    check.equal(str(original), str(dest))


def test_move2() -> None:
    original = Shape.from_ass_string("m 3636.335 1219.637 l 331.692 127.136 67.439 3781.479 1192.119 1322.969 2687.91 2197.805 -118.881 3049.314 -2758.659 -1776.605")
    dest = Shape.from_ass_string("m 3651.045 -776558.141 l 346.402 -777650.642 82.149 -773996.299 1206.829 -776454.809 2702.62 -775579.973 -104.171 -774728.464 -2743.949 -779554.383")
    original.move(math.pi * 80 / 1.5 ** 7, -777777.77777)
    original.round()
    check.equal(str(original), str(dest))


def test_scale1() -> None:
    original = Shape.from_ass_string("m -100.5 0 l 100 0 b 100 100 -100 100 -100.5 0 c")
    dest = Shape.from_ass_string("m -10.05 0 l 10 0 b 10 200 -10 200 -10.05 0 c")
    original.scale(1 / 10, 2)
    check.equal(str(original), str(dest))


def test_bounding1() -> None:
    original = Shape.from_ass_string("m -100.5 0 l 100 0 b 100 100 -100 100 -100.5 0 c")
    check.equal(
        tuple(tuple(p) for p in original.bounding),
        tuple(tuple(p) for p in (PC2D(-100.5, 0), PC2D(100, 100)))
    )


def test_bounding2() -> None:
    original = Shape.from_ass_string("m 0 0 l 20 0 20 10 0 10")
    check.equal(
        tuple(tuple(p) for p in original.bounding),
        tuple(tuple(p) for p in (PC2D(0.0, 0.0), PC2D(20.0, 10.0)))
    )


def test_align1() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m 0 -293.34 l 8 -18.67 829.33 0 768 -218.67 581.33 -296 280 -98.67')
    original.align(1)
    original.round()
    check.equal(str(original), str(dest))


def test_align2() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -414.665 -293.34 l -406.665 -18.67 414.665 0 353.335 -218.67 166.665 -296 -134.665 -98.67')
    original.align(2)
    original.round()
    check.equal(str(original), str(dest))


def test_align3() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -829.33 -293.34 l -821.33 -18.67 0 0 -61.33 -218.67 -248 -296 -549.33 -98.67')
    original.align(3)
    original.round()
    check.equal(str(original), str(dest))


def test_align4() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m 0 -145.34 l 8 129.33 829.33 148 768 -70.67 581.33 -148 280 49.33')
    original.align(4)
    original.round()
    check.equal(str(original), str(dest))


def test_align5() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -414.665 -145.34 l -406.665 129.33 414.665 148 353.335 -70.67 166.665 -148 -134.665 49.33')
    original.align(5)
    original.round()
    check.equal(str(original), str(dest))


def test_align6() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -829.33 -145.34 l -821.33 129.33 0 148 -61.33 -70.67 -248 -148 -549.33 49.33')
    original.align(6)
    original.round()
    check.equal(str(original), str(dest))


def test_align7() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m 0 2.66 l 8 277.33 829.33 296 768 77.33 581.33 0 280 197.33')
    original.align(7)
    original.round()
    check.equal(str(original), str(dest))


def test_align8() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -414.665 2.66 l -406.665 277.33 414.665 296 353.335 77.33 166.665 0 -134.665 197.33')
    original.align(8)
    original.round()
    check.equal(str(original), str(dest))


def test_align9() -> None:
    original = Shape.from_ass_string('m 58.67 77.33 l 66.67 352 888 370.67 826.67 152 640 74.67 338.67 272')
    dest = Shape.from_ass_string('m -829.33 2.66 l -821.33 277.33 0 296 -61.33 77.33 -248 0 -549.33 197.33')
    original.align(9)
    original.round()
    check.equal(str(original), str(dest))


def test_rotate_x() -> None:
    original = Shape.from_ass_string('m 336 -733.34 l 0 0 1722.66 -29.34 1066.66 -546.67')
    dest = Shape.from_ass_string('m 182.416 -371.715 l 0 0 1666.523 -26.5 655.346 -313.582')
    original.rotate(20.99, CartesianAxis.X)
    original.round()
    check.equal(str(original), str(dest))


def test_rotate_y() -> None:
    original = Shape.from_ass_string('m 336 -733.34 l 0 0 1722.66 -29.34 1066.66 -546.67')
    dest = Shape.from_ass_string('m 363.098 -794.516 l 0 0 2838.969 -48.477 1408.124 -723.524')
    original.rotate(-4.1, CartesianAxis.Y)
    original.round()
    check.equal(str(original), str(dest))


def test_rotate_z() -> None:
    original = Shape.from_ass_string('m 336 -733.34 l 0 0 1722.66 -29.34 1066.66 -546.67')
    dest = Shape.from_ass_string('m 518.679 -617.783 l 0 0 1668.6 429.176 1173.534 -243.784')
    original.rotate(15.4, CartesianAxis.Z)
    original.round()
    check.equal(str(original), str(dest))


def test_shear1() -> None:
    original = Shape.from_ass_string('m 576 1056 l 1000 1824 1788 1232 2000 1944 2500 1296 2480 1896 2904 936 b 2642 718 2230 918 2224 1296 2263 943 2053 797 1724 812 1500 1144 1368 1456 1160 1488 941 1547 1135 645 784 616')
    dest = Shape.from_ass_string('m 1315.2 940.8 l 2276.8 1624 2650.4 874.4 3360.8 1544 3407.2 796 3807.2 1400 3559.2 355.2 b 3144.6 189.6 2872.6 472 3131.2 851.2 2923.1 490.4 2610.9 386.4 2292.4 467.2 2300.8 844 2387.2 1182.4 2201.6 1256 2023.9 1358.8 1586.5 418 1215.2 459.2')
    original.shear(0.7, -0.2)
    original.round()
    check.equal(str(original), str(dest))


def test_shear2() -> None:
    original = Shape.from_ass_string('m 84.57 96 l 107.43 276.57 189.71 157.71 269.71 274.29 288 77.71 249.14 82.29 253.71 176 192 93.71 118.86 201.14 112 96')
    dest = Shape.from_ass_string('m 468.57 518.85 l 1213.71 813.72 820.55 1106.26 1366.87 1622.84 598.84 1517.71 578.3 1327.99 957.71 1444.55 566.84 1053.71 923.42 795.44 496 656')
    original.shear(4, 5)
    original.round()
    check.equal(str(original), str(dest))


def test_close1() -> None:
    original = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984')
    dest = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984 168 732')
    original.close()
    check.equal(str(original), str(dest))


def test_close2() -> None:
    original = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984 168 732')
    dest = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984 168 732')
    original.close()
    check.equal(str(original), str(dest))


def test_unclose1() -> None:
    original = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984')
    dest = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984')
    original.close()
    check.equal(str(original), str(dest))


def test_unclose2() -> None:
    original = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984 168 732')
    dest = Shape.from_ass_string('m 168 732 l 260 1440 872 916 512 984')
    original.close()
    check.equal(str(original), str(dest))
