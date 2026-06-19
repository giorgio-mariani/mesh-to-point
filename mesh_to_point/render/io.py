from pathlib import Path
from typing import List

import numpy as np

from mesh_to_point.camera import CameraModel, CameraPose


def _convert_to_params(camera: CameraModel) -> List:
    model = camera.model
    # TODO: add check over supported models
    params = [camera.fx] if model.startswith("SIMPLE_") else [camera.fx, camera.fy]
    return params + [camera.cx, camera.cy]


def _rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec  # w, x, y, z


def write_cameras_txt(file_path: str | Path, cameras: List[CameraModel]) -> None:
    """Write a COLMAP ``cameras.txt`` file from a sequence of camera models.

    The function serialises a list of :class:`~mesh_to_point.camera.CameraModel`
    objects to the format expected by COLMAP.  Each camera entry contains the
    camera identifier, model name, image width and height, and a list of
    intrinsic parameters.

    Parameters
    ----------
    file_path:
            Path to the output ``cameras.txt`` file.
    cameras:
            A list of :class:`CameraModel` instances. This class includes
            intrinsic parameters.

    Notes
    -----
    * The function writes a header that matches the COLMAP specification:
        ``# Camera list with one line of data per camera:``
        ``#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]``
    * The ``PARAMS[]`` section contains the focal lengths and principal
        point coordinates.

    """
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")

        for camera in cameras:
            cam_id = camera.camera_id
            model = camera.model
            params_str = " ".join(map(lambda x: f"{x:.20}", _convert_to_params(camera)))
            f.write(f"{cam_id} {model} {camera.width} {camera.height} {params_str}\n")


def write_images_txt(file_path: str | Path, camera_poses: List[CameraPose]) -> None:
    """Write a COLMAP ``images.txt`` file from a list of camera poses.

    The function serialises a list of :class:`~mesh_to_point.camera.CameraPose`
    objects to the format expected by COLMAP.  Each pose contains the
    camera-to-world rotation matrix ``R`` and translation vector ``t``.
    The rotation is converted to a quaternion in the order ``(w, x, y, z)``.

    Parameters
    ----------
    file_path:
            Path to the output ``images.txt`` file.
    camera_poses:
            A list of :class:`CameraPose` instances.  Each instance must expose
            the attributes ``image_id``, ``R`` (a 3x3 ``numpy.ndarray``), ``t``
            (a 3-element array), and ``camera_id``.

    Notes
    -----
    * The function writes an empty ``POINTS2D`` line after each image
        entry.
    * The output follows the exact header format used by COLMAP:

        ``# Image list with two lines of data per image:``
        ``#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME``
        ``#   POINTS2D[] as (X, Y, POINT3D_ID)``
    """
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(camera_poses)}\n")
        for cam_pose in camera_poses:
            qw, qx, qy, qz = _rotmat2qvec(cam_pose.R)
            tx, ty, tz = cam_pose.t
            f.write(
                f"{cam_pose.image_id} {qw:f} {qx:f} {qy:f} {qz:f} "
                f"{tx:f} {ty:f} {tz:f} {cam_pose.camera_id} {cam_pose.image_id:04d}_rgba.png\n"  # TODO: find a better way to get name
            )
            f.write("\n")  # empty points2D line


def write_json(
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
