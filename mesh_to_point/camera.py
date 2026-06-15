from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *

import numpy as np


@dataclass
class Camera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def image_coords(self) -> np.ndarray:
        """
        :return: ([self.height, self.width, 2]).reshape(self.height * self.width, 2) image coordinates
        """

    @abstractmethod
    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, get the direction that
        corresponds to "depth" in an RGBD rendering.

        This may raise an exception if there is no "D" channel in the
        corresponding ViewData.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 3] array of normalized depth directions.
        """
        _ = coords
        raise NotImplementedError


@dataclass
class ProjectiveCamera(Camera):
    """
    A Camera implementation for a standard pinhole camera.

    The camera rays shoot away from the origin in the z direction, with the x
    and y directions corresponding to the positive horizontal and vertical axes
    in image space.
    """

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    width: int
    height: int
    x_fov: float
    y_fov: float

    def image_coords(self) -> np.ndarray:
        ind = np.arange(self.width * self.height)
        coords = np.stack([ind % self.width, ind // self.width], axis=1).astype(np.float32)
        return coords

    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        # Normalize coordinates between -1 and +1 (both in the x and y axes)
        fracs = (coords / (np.array([self.width, self.height], dtype=np.float32) - 1)) * 2 - 1

        # Apply focal length transform?
        fracs = fracs * np.tan(np.array([self.x_fov, self.y_fov]) / 2)
        directions = self.z + self.x * fracs[:, :1] + self.y * fracs[:, 1:]
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        return np.stack([np.broadcast_to(self.origin, directions.shape), directions], axis=1)

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        return np.tile((self.z / np.linalg.norm(self.z))[None], [len(coords), 1])
