from dataclasses import dataclass
from pathlib import Path
from typing import *
import numpy as np
import json

from mesh_to_point.camera import ProjectiveCamera


@dataclass
class FrameData:
    camera: ProjectiveCamera
    depth_image: np.ndarray  # h*w x 1
    rgb_image: Optional[np.ndarray]  # h*w x 3
    alpha_image: Optional[np.ndarray]  # h*w x 1


def parse_data_directory(directory: Path) -> Generator[FrameData, None, None]:
    with open(directory / "transforms.json") as fp:
        data = json.load(fp)

    camera_model = data["camera_model"]
    for frame_data in data["frames"]:

        tmatrix = np.array(frame_data["transform_matrix"])
        camera = load_camera(
            camera_type=camera_model,
            origin=np.array(tmatrix[:3, 3]),
            x=np.array(tmatrix[:3, 0]),
            y=np.array(tmatrix[:3, 1]) * -1,  # Invert y-axis for camera coordinates
            z=np.array(tmatrix[:3, 2]) * -1,  # Invert z-axis for camera coordinates
            width=data["w"],
            height=data["h"],
            focal_length_x=data["fl_x"],
            focal_length_y=data["fl_y"],
        )

        depth_data = load_depth_file(directory / frame_data["depth_file_path"])

        if "file_path" in frame_data:
            rgb_data, alpha_data = load_rgba_file(directory / frame_data["file_path"])
        else:
            rgb_data, alpha_data = None, None

        yield FrameData(
            camera,
            depth_image=depth_data,
            rgb_image=rgb_data,
            alpha_image=alpha_data,
        )


def create_pointcloud_from_multiview(directory: Path, use_color: bool) -> np.ndarray:

    all_3d_coords = []
    all_rgb_values = []
    directory = Path(directory)
    for fd in parse_data_directory(directory):

        # Create an array of integer (x, y) image coordinates for Camera methods.
        image_coords = fd.camera.image_coords()

        # Select subset of pixels that have meaningful depth/color.
        image_mask = fd.depth_image <= 20.0
        if fd.alpha_image is not None:
            image_mask = image_mask & (fd.alpha_image >= 0.5)

        image_mask = image_mask.reshape(-1)
        image_coords = image_coords[image_mask]
        depth_image = fd.depth_image[image_mask]

        # Use the depth and camera information to compute the coordinates corresponding to every visible pixel.
        camera_rays = fd.camera.camera_rays(image_coords)
        camera_origins = camera_rays[:, 0]
        camera_directions = camera_rays[:, 1]
        depth_directions = fd.camera.depth_directions(image_coords)

        ray_scales = depth_image / np.sum(camera_directions * depth_directions, axis=-1, keepdims=True)
        coords_3d = camera_origins + camera_directions * ray_scales
        all_3d_coords.append(coords_3d)

        if use_color:
            all_rgb_values.append(fd.rgb_image[image_mask])

    coords = np.concatenate(all_3d_coords, axis=0)
    if use_color:
        rgb_values = np.concatenate(all_rgb_values, axis=0)
    else:
        rgb_values = None
    return coords, rgb_values


def load_camera(
    camera_type: str,
    origin: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    width: int,
    height: int,
    focal_length_x: float,
    focal_length_y: float,
) -> ProjectiveCamera:

    assert camera_type == "PINHOLE"

    return ProjectiveCamera(
        origin=origin,
        x=x,
        y=y,
        z=z,
        height=height,
        width=width,
        x_fov=2 * np.arctan(width / 2 / focal_length_x),
        y_fov=2 * np.arctan(height / 2 / focal_length_y),
    )


def load_depth_file(filepath: str) -> np.ndarray:
    import OpenEXR

    (exr_part,) = OpenEXR.File(str(filepath)).parts
    return np.array(exr_part.channels["V"].pixels).astype(np.float32).reshape(-1, 1)


def load_rgba_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    rgba = np.array(Image.open(filepath)).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return rgb.reshape(-1, 3), alpha.reshape(-1, 1)
