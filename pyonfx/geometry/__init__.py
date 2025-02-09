"""Geometry submodule"""
from __future__ import annotations

from functools import reduce
from itertools import chain
from math import asin, ceil, comb, cos, degrees, dist, fsum, inf, radians, sin, sqrt
from typing import Any, Iterable, Sequence, Tuple, TypeVar, overload

from .._logging import logger
from ..misc import chunk, clamp_value, frange
from .cartesian import Cartesian2D, Cartesian3D, CartesianAxis
from .coordinates import Axis, Coordinates
from .point import (
    Point, PointCartesian2D, PointCartesian3D, PointCylindrical, PointPolar, PointSpherical,
    PointsView, PointT
)
from .polar import Cylindrical, Polar, PolarAxis, Spherical
from .vector import Vector, VectorCartesian2D, VectorCartesian3D, VectorCylindrical, VectorPolar, VectorSpherical

__all__ = [
    'Coordinates',
    'CartesianAxis', 'PolarAxis',
    'Point',
    'PointsView',
    'PointCartesian2D', 'PointCartesian3D',
    'PointPolar', 'PointCylindrical', 'PointSpherical',
    'VectorCartesian2D', 'VectorCartesian3D',
    'VectorPolar', 'VectorCylindrical', 'VectorSpherical',
    'Geometry'
]


_Cartesian2DT = TypeVar('_Cartesian2DT', bound=Cartesian2D)
_Cartesian3DT = TypeVar('_Cartesian3DT', bound=Cartesian3D)
_PolarT = TypeVar('_PolarT', bound=Polar)
_CylindricalT = TypeVar('_CylindricalT', bound=Cylindrical)
_SphericalT = TypeVar('_SphericalT', bound=Spherical)

_BézierCurve = tuple[PointCartesian2D, PointCartesian2D, PointCartesian2D, PointCartesian2D]
_AssBézierCurve = tuple[PointCartesian2D, PointCartesian2D, PointCartesian2D]


class Geometry:
    """Collection of geometric methods for Point and Vectors"""
    __slots__ = ()

    @overload
    @staticmethod
    def rotate(_o: _Cartesian2DT, /, rot: float, axis: None, zp: tuple[float, ...]) -> _Cartesian2DT:
        """
        Rotate given Cartesian2D Point or Vector in given rotation

        :param _o:          Point or vector
        :param rot:         Rotation in degrees
        :param zp:          Zero point where the rotation will be performed
        :return:            Point or vector rotated
        """
        ...

    @overload
    @staticmethod
    def rotate(_o: _Cartesian3DT, /, rot: float, axis: Axis, zp: tuple[float, ...]) -> _Cartesian3DT:
        """
        Rotate given Cartesian3D Point or Vector in given rotation in a given axis

        :param _o:          Point or vector
        :param rot:         Rotation in degrees
        :param axis:        Axis
        :param zp:          Zero point where the rotation will be performed
        :return:            Point or vector rotated
        """
        ...

    @overload
    @staticmethod
    def rotate(_o: _CylindricalT, /, rot: float, axis: None, zp: None) -> _CylindricalT:
        """
        Rotate given Cylindrical Point or Vector in given rotation

        :param _o:          Point or vector
        :param rot:         Rotation in degrees
        :return:            Point or vector rotated
        """
        ...

    @overload
    @staticmethod
    def rotate(_o: _PolarT, /, rot: float, axis: None, zp: None) -> _PolarT:
        """
        Rotate given Polar Point or Vector in given rotation

        :param _o:          Point or vector
        :param rot:         Rotation in degrees
        :return:            Point or vector rotated
        """
        ...

    @overload
    @staticmethod
    def rotate(_o: _SphericalT, /, rot: float, axis: Axis, zp: None) -> _SphericalT:
        """
        Rotate given Spherical Point or Vector in given rotation in a given axis

        :param _o:          Point or vector
        :param rot:         Rotation in degrees
        :param axis:        Axis
        :return:            Point or vector rotated
        """
        ...

    @overload
    @staticmethod
    def rotate(_o: PointT, /, rot: float, axis: Axis, zp: tuple[float, ...] | None) -> PointT:
        ...

    @staticmethod
    @logger.catch(force_exit=True)
    def rotate(_o: Any, rot: float, axis: Any, zp: Any) -> Any:
        """Implementaton"""
        if isinstance(_o, Cartesian3D):
            _o.__rotate__(rot, axis, zp if zp else (0, 0, 0))
        elif isinstance(_o, Cartesian2D):
            _o.__rotate__(rot, zp=zp if zp else (0, 0))
        elif isinstance(_o, Spherical):
            _o.__rotate__(rot, axis)
        elif isinstance(_o, (Polar, Cylindrical)):
            _o.__rotate__(rot)
        else:
            raise NotImplementedError
        return _o

    @overload
    @staticmethod
    def vector(p0: PointCartesian2D, p1: PointCartesian2D) -> VectorCartesian2D:
        """
        Make a vector fom two PointCartesian2D

        :param p0:          First point
        :param p1:          Second point
        :return:            Vector
        """
        ...

    @overload
    @staticmethod
    def vector(p0: PointCartesian3D, p1: PointCartesian3D) -> VectorCartesian3D:
        """
        Make a vector fom two PointCartesian3D

        :param p0:          First point
        :param p1:          Second point
        :return:            Vector
        """
        ...

    @overload
    @staticmethod
    def vector(p0: PointPolar, p1: PointPolar) -> VectorPolar:
        """
        Make a vector fom two PointPolar

        :param p0:          First point
        :param p1:          Second point
        :return:            Vector
        """
        ...

    @overload
    @staticmethod
    def vector(p0: PointCylindrical, p1: PointCylindrical) -> VectorCylindrical:
        """
        Make a vector fom two PointCylindrical

        :param p0:          First point
        :param p1:          Second point
        :return:            Vector
        """
        ...

    @overload
    @staticmethod
    def vector(p0: PointSpherical, p1: PointSpherical) -> VectorSpherical:
        """
        Make a vector fom two PointSpherical

        :param p0:          First point
        :param p1:          Second point
        :return:            Vector
        """
        ...

    @overload
    @staticmethod
    def vector(p0: PointT, p1: PointT) -> Vector:
        ...

    @staticmethod
    @logger.catch(force_exit=True)
    def vector(p0: Any, p1: Any) -> Any:
        """Implementaton"""
        return p0.__vector__(p1)

    @overload
    @staticmethod
    def angle(v0: VectorCartesian2D, v1: VectorCartesian2D) -> float:
        """
        Get angle in radians between two VectorCartesian2D

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Angle in radians
        """
        ...

    @overload
    @staticmethod
    def angle(v0: VectorCartesian3D, v1: VectorCartesian3D) -> float:
        """
        Get angle in radians between two VectorCartesian3D

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Angle in radians
        """
        ...

    @overload
    @staticmethod
    def angle(v0: VectorPolar, v1: VectorPolar) -> float:
        """
        Get angle in radians between two VectorPolar

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Angle in radians
        """
        ...

    @overload
    @staticmethod
    def angle(v0: VectorCylindrical, v1: VectorCylindrical) -> float:
        """
        Get angle in radians between two VectorCylindrical

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Angle in radians
        """
        ...

    @overload
    @staticmethod
    def angle(v0: VectorSpherical, v1: VectorSpherical) -> float:
        """
        Get angle in radians between two VectorSpherical

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Angle in radians
        """
        ...

    @staticmethod
    @logger.catch(force_exit=True)
    def angle(v0: Any, v1: Any) -> Any:
        """Implementaton"""
        return v0.__angle__(v1)

    @overload
    @staticmethod
    def orthogonal(v0: VectorCartesian2D, v1: VectorCartesian2D) -> float:
        """
        Get orthogonal vector of two VectorCartesian2D

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Scalar
        """
        ...

    @overload
    @staticmethod
    def orthogonal(v0: VectorCartesian3D, v1: VectorCartesian3D) -> VectorCartesian3D:
        """
        Get orthogonal vector of two VectorCartesian3D

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Orthogonal vector
        """
        ...

    @overload
    @staticmethod
    def orthogonal(v0: VectorPolar, v1: VectorPolar) -> float:
        """
        Get orthogonal vector of two VectorPolar

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Scalar
        """
        ...

    @overload
    @staticmethod
    def orthogonal(v0: VectorCylindrical, v1: VectorCylindrical) -> VectorCylindrical:
        """
        Get orthogonal vector of two VectorCylindrical

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Orthogonal vector
        """
        ...

    @overload
    @staticmethod
    def orthogonal(v0: VectorSpherical, v1: VectorSpherical) -> VectorSpherical:
        """
        Get orthogonal vector of two VectorSpherical

        :param v0:          First vector
        :param v1:          Second vector
        :return:            Orthogonal vector
        """
        ...

    @staticmethod
    @logger.catch(force_exit=True)
    def orthogonal(v0: Any, v1: Any) -> Any:
        """Implementaton"""
        return v0.__orthogonal__(v1)

    @overload
    @staticmethod
    def stretch(v: VectorCartesian2D, /, length: float) -> VectorCartesian2D:
        """
        Scale VectorCartesian2D to given length

        :param v:           Vector
        :param length:      Required length
        :return:            Vector stretched
        """
        ...

    @overload
    @staticmethod
    def stretch(v: VectorCartesian3D, /, length: float) -> VectorCartesian3D:
        """
        Scale VectorCartesian3D to given length

        :param v:           Vector
        :param length:      Required length
        :return:            Vector stretched
        """
        ...

    @overload
    @staticmethod
    def stretch(v: VectorPolar, /, length: float) -> VectorPolar:
        """
        Scale VectorPolar to given length

        :param v:           Vector
        :param length:      Required length
        :return:            Vector stretched
        """
        ...

    @overload
    @staticmethod
    def stretch(v: VectorCylindrical, /, length: float) -> VectorCylindrical:
        """
        Scale VectorCylindrical to given length

        :param v:           Vector
        :param length:      Required length
        :return:            Vector stretched
        """
        ...

    @overload
    @staticmethod
    def stretch(v: VectorSpherical, /, length: float) -> VectorSpherical:
        """
        Scale VectorSpherical to given length

        :param v:           Vector
        :param length:      Required length
        :return:            Vector stretched
        """
        ...

    @staticmethod
    @logger.catch(force_exit=True)
    def stretch(v: Any, length: float) -> Any:
        """Implementation"""
        return v.__stretch__(length)

    @classmethod
    @logger.catch(force_exit=True)
    def line_intersect(
        cls,
        p0: PointCartesian2D, p1: PointCartesian2D,
        p2: PointCartesian2D, p3: PointCartesian2D,
        strict: bool = True
    ) -> PointCartesian2D:
        """
        Get line intersection coordinates between 4 points in the cartesian system

        :param p0:          First point
        :param p1:          Second point
        :param p2:          Third point
        :param p3:          Fourth point
        :param strict:      ???, defaults to True
        :return:            Point of the intersection
        """
        v0, v1 = cls.vector(p0, p1), cls.vector(p2, p3)

        if v0.norm * v1.norm == 0:
            raise ValueError(f'{cls.__name__}: lines mustn\'t have zero length')

        det = cls.orthogonal(v0, v1)

        if det == 0:
            return PointCartesian2D(inf, inf)

        def det_f(p0: Cartesian2D, p1: Cartesian2D) -> float:
            return p0.x * p1.y - p0.y * p1.x

        pre = det_f(p0, p1)
        post = det_f(p2, p3)
        ix, iy = (pre * v1.x - v0.x * post) / det, (pre * v1.y - v0.y * post) / det

        if strict:
            s = (ix - p1.x) / v0.x if v0.x != 0 else (iy - p1.y) / v0.y
            t = (ix - p3.x) / v1.x if v1.x != 0 else (iy - p3.y) / v1.y
            if s < 0 or s > 1 or t < 0 or t > 1:
                return PointCartesian2D(inf, inf)

        return PointCartesian2D(ix, iy)

    @staticmethod
    def split_line(p0: PointCartesian2D, p1: PointCartesian2D, max_length: float) -> list[PointCartesian2D]:
        """
        Split a line (p0, p1) in cartesian system into shorter lines with maximum max_length

        :param p0:          First point
        :param p1:          Second point
        :param max_length:  Maximum length
        :return:            List of new points
        """
        ncoord: list[PointCartesian2D] = []
        distance = dist(p0, p1)
        if distance > max_length:
            # Equal step between the two points instead of having all points to max_length
            # except the last for the remaining distance
            step = distance / ceil(distance / max_length)
            # Step can be a float so we're using frange
            for i in frange(step, distance, step):
                pct = i / distance
                ncoord.append(
                    PointCartesian2D(p0.x + (p1.x - p0.x) * pct, p0.y + (p1.y - p0.y) * pct)
                )
        ncoord.append(p1)
        return ncoord

    @classmethod
    def curve4_to_lines(cls, b_coord: _BézierCurve, tolerance: float, /) -> list[PointCartesian2D]:
        """
        Convert 4th degree curve to line points

        :param b_coord:     Bézier curve
        :param tolerance:   Tolerance in degrees
        :return:            List of PointCartesian2D
        """
        P = PointCartesian2D
        V = VectorCartesian2D

        ncoord: list[PointCartesian2D] = []
        tolerance = radians(tolerance)

        def _curve4_subdivide(b_coord: _BézierCurve, /) -> tuple[_BézierCurve, _BézierCurve]:
            """4th degree curve subdivider (De Casteljau)"""
            # Calculate points on curve vectors
            lcoord = list(chain.from_iterable(b_coord))
            sub3 = [sum(c) / 2 for c in zip(lcoord, lcoord[2:])]
            sub2 = [sum(c) / 2 for c in zip(sub3, sub3[2:])]
            subx1, suby1 = [sum(c) / 2 for c in zip(sub2, sub2[2:])]

            # Return new 2 curves
            b0 = b_coord[0], P(sub3[0], sub3[1]), P(sub2[0], sub2[1]), P(subx1, suby1)
            b1 = P(subx1, suby1), P(sub2[-2], sub2[-1]), P(sub3[-2], sub3[-1]), b_coord[-1]
            return b0, b1

        def _curve4_is_flat(b_coord: _BézierCurve, /) -> bool:
            """Check flatness of 4th degree curve with angles"""
            lcoord = list(chain.from_iterable(b_coord))
            # Pack curve vectors (only ones non zero)
            vecs = [
                reduce(lambda a, b: b - a, coord)
                for coord in zip(lcoord, lcoord[2:])
            ]
            vecsp = [
                (v0, v1) for v0, v1 in chunk(vecs, 2)
                if not (v0 == 0 and v1 == 0)
            ]
            # print(vecs)
            # print(vecsp)
            # # Old code:
            # (x0, y0), (x1, y1), (x2, y2), (x3, y3) = b_coord
            # vecs = [[x1 - x0, y1 - y0], [x2 - x1, y2 - y1], [x3 - x2, y3 - y2]]
            # vecsp = [el for el in vecs if not (el[0] == 0 and el[1] == 0)]

            # Check flatness on vectors
            for v0, v1 in zip(vecsp, vecsp[1:]):
                if abs(cls.angle(V(*v0), V(*v1))) > tolerance:
                    return False
            return True

        def _convert_recursive(b_coord: _BézierCurve, /) -> None:
            """Conversion in recursive processing"""
            if _curve4_is_flat(b_coord):
                ncoord.append(b_coord[-1])
                return None
            b0, b1 = _curve4_subdivide(b_coord)
            _convert_recursive(b0)
            _convert_recursive(b1)

        # Splitting curve recursively until we're not satisfied (angle <= tolerance)
        _convert_recursive(b_coord)
        return ncoord

    @staticmethod
    def point_on_segment(p0: PointCartesian2D, p1: PointCartesian2D, factor: float = 0.5) -> PointCartesian2D:
        """
        Calculate new point on the line segment formed by p0 and p1

        :param p0:          First point
        :param p1:          Second point
        :param factor:      Factor parameter where the point is, defaults to 0.5
        :return:            Point in this line segment
        """
        return PointCartesian2D(
            p0.x + factor * (p1.x - p0.x),
            p0.y + factor * (p1.y - p0.y)
        )

    @staticmethod
    def point_on_bézier_curve(curv: Sequence[Point], factor: float = 0.5, *, use_fsum: bool = False) -> PointCartesian3D:
        """
        Calculate the coordinates of a point on a Bézier curve

        :param curv:        List of Point representing a Bézier curve of any degree n.
        :param factor:      Factor parameter where the point is, defaults to 0.5
        :param use_fsum:    If True use math.fnum otherwise ``sum`` python built-in
        :return:            Point in this Bézier curve
        """
        n = len(curv) - 1

        def calc_c(c: float, i: int) -> float:
            return comb(n, i) * (1 - factor) ** (n - i) * factor ** i * c

        def _sum(_seq: Iterable[float]) -> float:
            return fsum(_seq) if use_fsum else sum(_seq)

        return PointCartesian3D(
            *[_sum(calc_c(v, i)
              for i, v in enumerate(coord_zip))
              for coord_zip in zip(*(p.to_3d() for p in curv))]
        )

    @classmethod
    def round_vertex(
        cls,
        p0: PointCartesian2D, p1: PointCartesian2D, p2: PointCartesian2D,
        deviation: float, tolerance: float = 157.5, tension: float = 0.5
    ) -> list[PointCartesian2D]:
        """
        Round vertex in a cubic bézier curve

        :param p0:          First point
        :param p1:          Second point
        :param p2:          Third point
        :param deviation:   Length in pixel of the deviation from each vertex
        :param tolerance:   Angle in degree to define a vertex to be rounded.
                            If the vertex's angle is lower than tolerance then it will be rounded.
                            Valid ranges are 0.0 - 180.0, defaults to 157.5
        :param tension:     Adjust point tension in percentage, defaults to 0.5
        :return:            List of one point if the angle is lower than tolerance else 4 points
        """
        v0 = cls.vector(p0, p1)
        v1 = cls.vector(p2, p1)
        tension = clamp_value(tension, 0., 1.)
        if degrees(cls.angle(v0, v1)) < tolerance:
            try:
                b0 = cls.point_on_segment(p1, p0, clamp_value(deviation / v0.norm, 0., 1.))
            except ZeroDivisionError:
                b0 = p1
            try:
                b3 = cls.point_on_segment(p1, p2, clamp_value(deviation / v1.norm, 0., 1.))
            except ZeroDivisionError:
                b3 = p1
            b1 = cls.point_on_segment(p1, b0, tension)
            b2 = cls.point_on_segment(p1, b3, tension)
            return [b0, b1, b2, b3]
        return [p1]

    @staticmethod
    def make_ellipse(
        w: float, h: float,
        c_xy: tuple[float, float] = (0., 0.), /, clockwise: bool = True
    ) -> Tuple[
        PointCartesian2D,
        _AssBézierCurve, _AssBézierCurve, _AssBézierCurve, _AssBézierCurve
    ]:
        """
        Make ellipse coordinates with given width and height, centered around (c_xy)

        :param w:               Width of the ellipse
        :param h:               Height of the ellipse
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                Ellipses coordinates
        """
        c = 0.551915024494  # https://spencermortensen.com/articles/bezier-circle/
        cx, cy = c_xy
        cl = - int((-1) ** clockwise)
        P = PointCartesian2D
        return (
            P((cx - 0) * cl, cy + h),
            (
                P((cx - w * c) * cl, cy + h),
                P((cx - w) * cl, cy + h * c),
                P((cx - w) * cl, cy - 0)
            ),
            (
                P((cx - w) * cl, cy - h * c),
                P((cx - w * c) * cl, cy - h),
                P((cx - 0) * cl, cy - h)
            ),
            (
                P((cx + w * c) * cl, cy - h),
                P((cx + w) * cl, cy - h * c),
                P((cx + w) * cl, cy - 0)
            ),
            (
                P((cx + w) * cl, cy + h * c),
                P((cx + w * c) * cl, cy + h),
                P((cx - 0) * cl, cy + h)
            )
        )

    @staticmethod
    def make_parallelogram(
        w: float, h: float, angle: float,
        c_xy: tuple[float, float] = (0., 0.), /, clockwise: bool = True
    ) -> tuple[PointCartesian2D, PointCartesian2D, PointCartesian2D, PointCartesian2D, PointCartesian2D]:
        """
        Make parallelogram coordinates with given width, height and angle, centered around (c_xy)

        :param w:               Width of the parallelogram
        :param h:               Height of the parallelogram
        :param angle:           First angle of the parallelogram in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :return:                Parallelogram coordinates
        """
        cl = - int((-1) ** clockwise)
        cx, cy = c_xy

        l = h / cos(radians(90 - angle))
        x0, y0 = 0, 0
        x1, y1 = l * cos(radians(angle)), l * sin(radians(angle))
        x2, y2 = x1 + w, y1
        x3, y3 = w, 0

        P = PointCartesian2D

        return (
            P((x0 + cx) * cl, y0 + cy),
            P((x1 + cx) * cl, y1 + cy),
            P((x2 + cx) * cl, y2 + cy),
            P((x3 + cx) * cl, y3 + cy),
            P((x0 + cx) * cl, y0 + cy)
        )

    @classmethod
    @logger.catch(force_exit=True)
    def make_triangle(
        cls,
        side: float | tuple[float, float], angle: tuple[float, float] | float,
        c_xy: tuple[float, float] = (0., 0.), /, clockwise: bool = True
    ) -> tuple[PointCartesian2D, PointCartesian2D, PointCartesian2D, PointCartesian2D]:
        """
        Make general triangle coordinates with given sides and angles, centered around (c_xy)

        :param side:            Side(s) of the triangle
        :param angle:           Angle(s) of the triangle in degrees
        :param c_xy:            Center (x, y) coordinate, defaults to (0., 0.)
        :param clockwise:       Direction of point creation, defaults to True
        :param orthocentred:    Centred in the orthocenter, defaults to True
        :return:                Triangle coordinates
        """
        cl = - int((-1) ** clockwise)
        cx, cy = c_xy
        P = PointCartesian2D

        if isinstance(side, (int, float)) and isinstance(angle, tuple):
            A, B = angle
            C = 180 - A - B
            ab = side
            bc = sin(radians(A)) * ab / sin(radians(C))

            x0, y0 = 0, 0
            x1, y1 = ab, 0
            x2, y2 = x1 - bc * cos(radians(B)), bc * sin(radians(B))
        elif isinstance(side, tuple) and isinstance(angle, (int, float)):
            ab, ac = side
            A = angle
            bc = sqrt(ac ** 2 + ab ** 2 - 2 * ac * ab * cos(radians(A)))
            Br = asin(sin(radians(A)) * ac / bc)

            x0, y0 = 0, 0
            x1, y1 = ab, 0
            x2, y2 = x1 - bc * cos(Br), bc * sin(Br)
        else:
            raise ValueError(
                f'{cls.__name__}: possibles values are one side and two angles '
                + 'or two sides and one angle'
            )

        return (
            P((x0 + cx) * cl, y0 + cy),
            P((x1 + cx) * cl, y1 + cy),
            P((x2 + cx) * cl, y2 + cy),
            P((x0 + cx) * cl, y0 + cy)
        )
