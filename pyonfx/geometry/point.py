"""Point interface"""

from __future__ import annotations

from abc import ABC, abstractmethod
from cmath import phase, polar, rect
from math import atan2, cos, radians, sin, sqrt
from typing import TypeVar
from typing import cast as typing_cast

import numpy as np
from numpy.typing import NDArray

from .._logging import logger
from ..ptypes import View
from .cartesian import Cartesian2D, Cartesian3D
from .coordinates import Coordinates
from .polar import Cylindrical, Polar, Spherical
from .vector import Vector, VectorCartesian2D, VectorCartesian3D, VectorCylindrical, VectorPolar, VectorSpherical

PointT = TypeVar('PointT', bound='Point')


class Point(Coordinates, ABC, empty_slots=True):
    """Abstract Point interface with abstract __vector__ method"""

    @abstractmethod
    def __vector__(self: PointT, p: PointT) -> Vector:
        """
        Get the vector of this point and the given point

        :param p:       Another point
        :return:        Vector
        """
        ...

    @abstractmethod
    def to_2d(self) -> PointCartesian2D:
        ...

    @abstractmethod
    def to_3d(self) -> PointCartesian3D:
        ...

    @abstractmethod
    def to_polar(self) -> PointPolar:
        ...

    @abstractmethod
    def to_cylindrical(self) -> PointCylindrical:
        ...

    @abstractmethod
    def to_spherical(self) -> PointSpherical:
        ...


class PointCartesian2D(Cartesian2D, Point):
    """Point in Cartesian2D system"""

    def __vector__(self, p: PointCartesian2D) -> VectorCartesian2D:
        return VectorCartesian2D(self.x - p.x, self.y - p.y)

    def to_2d(self) -> PointCartesian2D:
        return self

    def to_3d(self) -> PointCartesian3D:
        return PointCartesian3D(self.x, self.y, 0.)

    def to_polar(self) -> PointPolar:
        return PointPolar(*polar(complex(*self)))

    def to_cylindrical(self) -> PointCylindrical:
        return self.to_polar().to_cylindrical()

    def to_spherical(self) -> PointSpherical:
        return self.to_polar().to_spherical()

    def as_vector(self, *, cast: bool = False) -> VectorCartesian2D:
        return typing_cast(VectorCartesian2D, self) if cast else VectorCartesian2D(*self)


class PointCartesian3D(Cartesian3D, Point):
    """Point in Cartesian3D system"""

    def __vector__(self, p: PointCartesian3D) -> VectorCartesian3D:
        return VectorCartesian3D(self.x - p.x, self.y - p.y, self.z - p.z)

    def to_2d(self) -> PointCartesian2D:
        return PointCartesian2D(self.x, self.y)

    def to_3d(self) -> PointCartesian3D:
        return self

    def to_polar(self) -> PointPolar:
        return self.to_2d().to_polar()

    def to_cylindrical(self) -> PointCylindrical:
        return PointCylindrical(*polar(complex(self.x, self.y)), z=self.z)

    def to_spherical(self) -> PointSpherical:
        r, theta = polar(complex(self.x ** 2 + self.y ** 2, self.z))
        return PointSpherical(r, phase(complex(self.y, self.x)), theta)

    def as_vector(self, *, cast: bool = False) -> VectorCartesian3D:
        return typing_cast(VectorCartesian3D, self) if cast else VectorCartesian3D(*self)

    @logger.catch
    def project_2d(self) -> PointCartesian2D:
        """
        Project on two dimensions

        :return:                Projected Point
        """
        # https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        # Length of the camera seems to be 312 according to my tests
        import cv2  # type: ignore
        img_pts, _ = cv2.projectPoints(
            objectPoints=np.array(self, np.float64),
            rvec=np.zeros(3, np.float64),
            tvec=np.array((0, 0, -312), np.float64),
            cameraMatrix=np.array([(-312, 0, 0), (0, -312, 0), (0, 0, 1)], np.float64),
            distCoeffs=np.zeros((4, 1), np.float64),
        )
        img_pts = typing_cast(NDArray[np.float32], img_pts)
        return PointCartesian2D(*map(float, img_pts.flatten()))


class PointPolar(Polar, Point):
    """Point in Polar system"""

    def __vector__(self, p: PointPolar) -> VectorPolar:
        return VectorPolar(
            sqrt(self.r ** 2 + p.r ** 2 - 2 * self.r * p.r * cos(p.phi - self.phi)),
            self.phi - atan2(p.phi * sin(p.phi - self.phi), self.phi - p.phi * cos(p.phi - self.phi))
        )

    def to_2d(self) -> PointCartesian2D:
        co = rect(self.r, self.phi)
        return PointCartesian2D(co.real, co.imag)

    def to_3d(self) -> PointCartesian3D:
        return self.to_2d().to_3d()

    def to_polar(self) -> PointPolar:
        return self

    def to_cylindrical(self) -> PointCylindrical:
        return PointCylindrical(self.r, self.phi, 0)

    def to_spherical(self) -> PointSpherical:
        return PointSpherical(self.r, self.phi, radians(90.))


class PointCylindrical(Cylindrical, Point):
    """Point in Cylindrical system"""

    def __vector__(self, p: PointCylindrical) -> VectorCylindrical:
        return VectorCylindrical(
            sqrt(self.r ** 2 + p.r ** 2 - 2 * self.r * p.r * cos(p.phi - self.phi)),
            self.phi - atan2(p.phi * sin(p.phi - self.phi), self.phi - p.phi * cos(p.phi - self.phi)),
            self.z - p.z
        )

    def to_2d(self) -> PointCartesian2D:
        co = rect(self.r, self.phi)
        return PointCartesian2D(co.real, co.imag)

    def to_3d(self) -> PointCartesian3D:
        co = rect(self.r, self.phi)
        return PointCartesian3D(co.real, co.imag, self.z)

    def to_polar(self) -> PointPolar:
        return PointPolar(self.r, self.phi)

    def to_cylindrical(self) -> PointCylindrical:
        return self

    def to_spherical(self) -> PointSpherical:
        r, theta = polar(complex(self.r, self.z))
        return PointSpherical(r, self.phi, theta)


class PointSpherical(Spherical, Point):
    """Point in Spherical system"""

    def __vector__(self, p: PointSpherical) -> VectorSpherical:
        return self.to_3d().__vector__(p.to_3d()).to_spherical()

    def to_2d(self) -> PointCartesian2D:
        return self.to_3d().to_2d()

    def to_3d(self) -> PointCartesian3D:
        return PointCartesian3D(
            self.r * sin(self.theta) * cos(self.phi),
            self.r * sin(self.theta) * sin(self.phi),
            self.r * cos(self.theta)
        )

    def to_polar(self) -> PointPolar:
        return PointPolar(
            self.r * sin(self.theta),
            self.phi,
        )

    def to_cylindrical(self) -> PointCylindrical:
        return PointCylindrical(
            self.r * sin(self.theta),
            self.phi,
            self.r * cos(self.theta)
        )

    def to_spherical(self) -> PointSpherical:
        return self


class PointsView(View[Point]):
    """View for points"""
    ...
