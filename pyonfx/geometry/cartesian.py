"""Coordinates cartesian system interface"""

from abc import ABC
from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from ..ptypes import NamedMutableSequence
from .coordinates import Axis, Coordinates


class CartesianAxis(Axis):
    """Simple Enum for Cartesian axis"""
    X = 0
    Y = 1
    Z = 2


class Cartesian2D(Coordinates, ABC, empty_slots=True):
    """Cartesian coordinate system in two dimensions"""

    x: float
    """Abscissa coordinate"""

    y: float
    """Ordinate coordinate"""

    def __init__(self, x: float, y: float) -> None:
        """
        Make a new object in the 2D cartesian system

        :param _x:              Abscissa
        :param _y:              Ordinate
        """
        # Faster access than using super()
        NamedMutableSequence.__init__(self, x=x, y=y)

    def __rotate__(self, rot: float, axis: CartesianAxis = CartesianAxis.Z, zp: tuple[float, ...] = (0., 0.)) -> None:
        """
        Rotate on an imaginary Z-axis

        :param rot:             Needed rotation in degrees
        :param zp:              Zero point where the rotation will be performed, defaults to (0, 0)
        """
        theta = radians(rot)
        R = np.array([(cos(theta), -sin(theta)), (sin(theta), cos(theta))], np.float32)  # type: ignore[var-annotated]
        O, P = np.atleast_2d(np.asanyarray(zp, np.float32)), np.atleast_2d(np.asanyarray(self, np.float32))  # type: ignore[var-annotated]
        for attr, value in zip(self.__slots__, np.squeeze((R @ (P.T - O.T) + O.T).T)):
            setattr(self, attr, float(value))


class Cartesian3D(Cartesian2D, ABC, empty_slots=True):
    """Cartesian coordinate system in three dimensions"""

    z: float
    """Applicate coordinate"""

    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Make a new object in the 3D cartesian system

        :param _x:              Abscissa
        :param _y:              Ordinate
        :param _z:              Applicate
        """
        # Faster access than using super()
        NamedMutableSequence.__init__(self, x=x, y=y, z=z)

    def __rotate__(self, rot: float, axis: CartesianAxis = CartesianAxis.Z, zp: tuple[float, ...] = (0., 0., 0.)) -> None:
        """
        Rotate on given axis

        :param rot:             Needed rotation in degrees
        :param axis:            The rotation will be performed on this axis
        :param zp:              Zero point where the rotation will be performed, defaults to (0, 0, 0)
        """
        rot_mats = (_rot_mat_x, _rot_mat_y, _rot_mat_z)
        try:
            rmat = rot_mats[axis]
        except IndexError as i_err:
            raise ValueError(f'{self.__class__.__name__}: Wrong axis number') from i_err

        R = rmat(radians(rot))
        O = np.atleast_3d(np.asanyarray(zp, np.float32))  # type: ignore[var-annotated]
        P = np.atleast_3d(np.asanyarray(self, np.float32))  # type: ignore[var-annotated]

        for attr, value in zip(self.__slots__, np.squeeze((R @ (P.T - O.T) + O.T).T)):
            setattr(self, attr, float(value))


def _rot_mat_x(theta: float) -> NDArray[np.float32]:
    return np.array(
        [(1, 0, 0),
         (0, cos(theta), -sin(theta)),
         (0, sin(theta), cos(theta))], np.float32
    )


def _rot_mat_y(theta: float) -> NDArray[np.float32]:
    return np.array(
        [(cos(theta), 0, sin(theta)),
         (0, 1, 0),
         (-sin(theta), 0, cos(theta))], np.float32
    )


def _rot_mat_z(theta: float) -> NDArray[np.float32]:
    return np.array(
        [(cos(theta), -sin(theta), 0),
         (sin(theta), cos(theta), 0),
         (0, 0, 1)], np.float32
    )
