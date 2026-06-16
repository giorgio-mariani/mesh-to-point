from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import *

import numpy as np

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraPose:
    image_id: int
    camera_id: int

    # World-to-camera transform, as stored natively by COLMAP:
    #   X_cam = R @ X_world + t
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @property
    def world_to_cam(self) -> np.ndarray:
        """4x4 world-to-camera matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def cam_to_world(self) -> np.ndarray:
        """4x4 camera-to-world matrix (common convention for NeRF, etc.)."""
        T = np.eye(4)
        R_c2w = self.R.T
        t_c2w = -R_c2w @ self.t
        T[:3, :3] = R_c2w
        T[:3, 3] = t_c2w
        return T

    @property
    def camera_center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t


@dataclass
class CameraModel:
    camera_id: int
    model: str  # e.g. "PINHOLE", "SIMPLE_PINHOLE", "OPENCV"
    width: int
    height: int
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic calibration matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def image_coords(self) -> np.ndarray:
        ind = np.arange(self.width * self.height)
        coords = np.stack([ind % self.width, ind // self.width], axis=1)
        return coords.astype(np.float32)

    def camera_rays(self, camera_pose: CameraPose, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """
        x, y, z = camera_pose.R.T
        x_fov = 2 * math.atan(self.width / (2 * self.fx))
        y_fov = 2 * math.atan(self.height / (2 * self.fy))

        # Normalize coordinates between -1 and +1 (both in the x and y axes)
        fracs = (
            coords / (np.array([self.width, self.height], dtype=np.float32) - 1)
        ) * 2 - 1

        fracs = fracs * np.tan(np.array([x_fov, y_fov]) / 2)
        directions = z + x * fracs[:, :1] + y * fracs[:, 1:]
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        return np.stack(
            [np.broadcast_to(self.origin, directions.shape), directions], axis=1
        )

    def depth_directions(
        self, camera_pose: CameraPose, coords: np.ndarray
    ) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, get the direction that
        corresponds to "depth" in an RGBD rendering.

        This may raise an exception if there is no "D" channel in the
        corresponding ViewData.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 3] array of normalized depth directions.
        """

        _, _, z = camera_pose.R.T
        return np.tile((z / np.linalg.norm(z))[None], [len(coords), 1])


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


def from_COLMAP(camera_file: Path) -> Tuple[CameraModel, List[CameraPose]]:
    """Read a camera_file in the COLMAP format."""
    ...
    # TODO


def from_nerfstudio(camera_file: Path) -> Tuple[CameraModel, List[CameraPose]]:
    with open(camera_file, "r") as fp:
        camera_data = json.load(fp)
        camera_id = 0

        camera_model = camera_data["camera_model"]
        # TODO: checkit is only pinhole or simple_pinhole

        camera_intrinsics = CameraModel(
            camera_id=camera_id,
            model=camera_data["camera_model"],
            width=camera_data["w"],
            height=camera_data["h"],
            fx=camera_data["fl_x"],
            fy=camera_data["fl_y"],
            cx=camera_data["cx"],
            cy=camera_data["cy"],
        )

        camera_extrinsics = []
        for img_id, p in enumerate(camera_data["frames"]):

            # TODO check that no intrinsic parameter redifinition occur only extrinsic are admissible
            # TODO add warning about paths not working

            transform_matrix = np.array(p["transform_matrix"])
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]

            cam_pose = CameraPose(
                camera_id=camera_id,
                image_id=img_id,
                R=R,
                t=t,
            )

            camera_extrinsics.append(cam_pose)
    return camera_intrinsics, camera_extrinsics
