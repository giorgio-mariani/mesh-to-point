from dataclasses import dataclass
import json
from typing import List
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

    def validate(self):
        """Validate the camera model and parameters.

        Raises
        ------
        ValueError
            If the camera model is not supported or if any of the intrinsic parameters are invalid.
        """

        if self.model not in SUPPORTED_CAMERA_MODELS:
            raise ValueError(
                f"Unsupported camera model '{self.model}'. Supported models are: "
                f"{', '.join(sorted(SUPPORTED_CAMERA_MODELS))}."
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive integers.")

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths fx and fy must be positive.")

        if not np.isclose(self.cx, self.width / 2):
            raise ValueError(
                f"Principal point cx={self.cx} is not at the image center (width/2={self.width / 2})."
            )

        if not np.isclose(self.cy, self.height / 2):
            raise ValueError(
                f"Principal point cy={self.cy} is not at the image center (height/2={self.height / 2})."
            )

        if not np.isclose(self.fx, self.fy):
            raise ValueError(f"focal lengths fx and fy must be equal.")

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


def read_camera_config(camera_file: str | Path) -> Tuple[CameraModel, List[CameraPose]]:
    """Read camera extrinsic and intrinsic parameters from a Nerfstudio ``transform.json`` file.

    The function parses a JSON file that follows the Nerfstudio camera configuration format
    (see :func:`write_camera_config`).  It extracts the intrinsic camera model and
    parameters, and builds a :class:`CameraModel` instance.  For each frame in the
    ``frames`` list it constructs a :class:`CameraPose` containing the camera‑to‑world
    transformation matrix.

    Parameters
    ----------
    camera_file : str | :class:`pathlib.Path`
        Path to the JSON file containing the camera configuration.  The file is
        expected to contain the keys ``camera_model``, ``w``, ``h``, ``fl_x``,
        ``fl_y``, ``cx`` and ``cy`` for the intrinsic block, and a list of
        ``frames`` where each frame contains a ``transform_matrix``.

    Returns
    -------
    Tuple[:class:`CameraModel`, List[:class:`CameraPose`]]
        A tuple where the first element is the intrinsic camera model and the
        second element is a list of camera poses (one per frame).  The order of
        the poses matches the order of the ``frames`` array in the JSON.
    """
    camera_file = Path(camera_file)
    with open(camera_file, "r") as fp:
        camera_data = json.load(fp)
        camera_id = 0

        _INTRINSIC_KEYS = {"camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy"}
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

        # Validate camera model type – only a few are supported by this code.
        camera_intrinsics.validate()

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


def write_camera_config(
    file_path: str | Path, camera: CameraModel, camera_poses: List[CameraPose]
) -> None:
    """Write camera parameters in the Nerfstudio ``transform.json`` format.

    The output JSON follows the structure used by the Nerfstudio pipeline:

    .. code-block:: json

        {
            "camera_model": "PINHOLE",
            "w": 1920,
            "h": 1080,
            "fl_x": 1000.0,
            "fl_y": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "frames": [
                {"transform_matrix": [[...], [...], [...], [...]]},
                ...
            ]
        }

    Parameters
    ----------
    file_path:
        Path to write the JSON file.
    camera:
        :class:`CameraModel` instance containing intrinsic parameters.
    camera_poses:
        List of :class:`CameraPose` objects providing extrinsics for each frame.
    """
    data: dict = {
        "camera_model": camera.model,
        "w": camera.width,
        "h": camera.height,
        "fl_x": camera.fx,
        "fl_y": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "frames": [],
    }

    for pose in camera_poses:
        # Nerfstudio expects a 4x4 camera‑to‑world matrix.
        matrix = pose.cam_to_world.tolist()
        data["frames"].append(
            {
                "file_path": f"images/{pose.camera_id:04}_rgba.png",
                "transform_matrix": matrix,
            }
        )

    # Write JSON with pretty formatting.
    import json

    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=4)
