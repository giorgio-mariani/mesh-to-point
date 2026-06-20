from dataclasses import dataclass
import json
import warnings
from pathlib import Path
from typing import *

import numpy as np

from dataclasses import dataclass
import numpy as np

SUPPORTED_CAMERA_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})


@dataclass
class CameraPose:
    image_id: int
    camera_id: int

    # camera-to-world transform
    #   X_world = R @ X_cam + t
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @property
    def cam_to_world(self) -> np.ndarray:
        """4x4 world-to-camera matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def world_to_cam(self) -> np.ndarray:
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
        """"""
        ind = np.arange(self.width * self.height)
        coords = np.stack(
            [ind % self.width, self.width - 1 - ind // self.width], axis=1
        )
        return coords.astype(np.float32)

    def camera_rays(
        self, camera_pose: CameraPose, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel in world space.

        Parameters
        ----------
        coords : np.ndarray
            Shape (N, 2) integer pixel coordinates.

        Returns
        -------
        origins : np.ndarray
            Shape (N, 3) world-space ray origins (camera centre).
        directions : np.ndarray
            Shape (N, 3) unit-length world-space ray directions.
        """
        # 1. Pixel → camera‑space direction
        #    (x - cx) / fx, (y - cy) / fy, 1
        u = (coords[:, 0] - self.cx) / self.fx
        v = (coords[:, 1] - self.cy) / self.fy
        dirs_cam = np.stack(
            [u, v, -np.ones_like(u)], axis=1
        )  # NOTE: We use -1 because we adopt Blender's convention of the negative z-axis as the camera forward

        # 2. Normalise
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)

        # 3. Camera → world
        R = camera_pose.cam_to_world[:3, :3]
        dirs_world = dirs_cam @ R.T  # (N,3)

        # 4. Ray origins (camera centre)
        origin = camera_pose.cam_to_world[:3, 3]  # (3,)
        origins = np.tile(origin, (len(coords), 1))

        return origins, dirs_world


def from_json(camera_file: str | Path) -> Tuple[CameraModel, List[CameraPose]]:
    camera_file = Path(camera_file)
    with open(camera_file, "r") as fp:
        camera_data = json.load(fp)
        camera_id = 0

        # Validate camera model type – only a few are supported by this code.
        cam_model = camera_data["camera_model"]
        if cam_model not in SUPPORTED_CAMERA_MODELS:
            raise ValueError(
                f"Unsupported camera model '{cam_model}'. Supported models are: "
                f"{', '.join(sorted(SUPPORTED_CAMERA_MODELS))}."
            )

        _INTRINSIC_KEYS = {"camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy"}

        camera_intrinsics = CameraModel(
            camera_id=camera_id,
            model=cam_model,
            width=camera_data["w"],
            height=camera_data["h"],
            fx=camera_data["fl_x"],
            fy=camera_data["fl_y"],
            cx=camera_data["cx"],
            cy=camera_data["cy"],
        )

        camera_extrinsics = []
        # Keys that belong to the intrinsic block – if any of these appear in a frame, we warn.
        for img_id, p in enumerate(camera_data["frames"]):
            # Detect accidental re‑definition of intrinsics
            if any(key in p for key in _INTRINSIC_KEYS):
                warnings.warn(
                    f"Frame {img_id} contains intrinsic parameters that will be ignored. "
                    "Only extrinsic parameters are admissible.",
                    RuntimeWarning,
                )

            if "file_path" in p:
                img_path = Path(camera_file.parent, p["file_path"])
                if not img_path.is_file():
                    warnings.warn(
                        f"Image file '{img_path}' referenced in frame {img_id}: "
                        f"This parameter will be ignored.",
                        RuntimeWarning,
                    )

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
