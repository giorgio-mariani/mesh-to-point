from dataclasses import dataclass
from pathlib import Path
from typing import *
import numpy as np
import json

from mesh_to_point.camera import CameraModel, CameraPose, from_nerfstudio
from mesh_to_point.pointcloud.misc import colorize_pointcloud, subsample_pointcloud


@dataclass
class ViewData:
    camera_model: CameraModel
    camera_pose: CameraPose
    depth_image: np.ndarray  # h*w x 1
    rgb_image: Optional[np.ndarray]  # h*w x 3
    alpha_image: Optional[np.ndarray]  # h*w x 1


def load_multiview_images(camera_file: str | Path) -> Generator[ViewData, None, None]:
    camera_file = Path(camera_file)

    camera_model, camera_poses = from_nerfstudio(camera_file)

    for pose in camera_poses:
        file_prefix = camera_file.parent / f"{pose.image_id:04d}"
        rgb_data, alpha_data = load_rgba_file(f"{file_prefix}_rgba.png")
        # TODO: check depth file exists
        depth_data = load_depth_file(f"{file_prefix}_depth.exr")

        yield ViewData(
            camera_model,
            depth_image=depth_data,
            rgb_image=rgb_data,
            alpha_image=alpha_data,
        )


def merge_multiviews(multiview_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:

    all_3d_coords = []
    all_rgb_values = []
    multiview_dir = Path(multiview_dir)
    for fd in load_multiview_images(multiview_dir / "transforms.json"):

        # Create an array of integer (x, y) image coordinates for Camera methods.
        image_coords = fd.camera_model.image_coords()

        # Select subset of pixels that have meaningful depth/color.
        image_mask = fd.depth_image <= 20.0
        if fd.alpha_image is not None:
            image_mask = image_mask & (fd.alpha_image >= 0.5)

        image_mask = image_mask.reshape(-1)
        image_coords = image_coords[image_mask]
        depth_values = fd.depth_image[image_mask]
        rgb_values = fd.rgb_image[image_mask]

        # Use the depth and camera information to compute the coordinates corresponding to every visible pixel.
        camera_rays = fd.camera_model.camera_rays(fd.camera_pose, image_coords)
        camera_origins, camera_directions = camera_rays[:, 0], camera_rays[:, 1]
        depth_directions = fd.camera_model.depth_directions(
            fd.camera_pose, image_coords
        )

        ray_scales = depth_values / np.sum(
            camera_directions * depth_directions, axis=-1, keepdims=True
        )
        coords_3d = camera_origins + camera_directions * ray_scales

        # Update cumulative data
        all_3d_coords.append(coords_3d)
        all_rgb_values.append(rgb_values)

    coords = np.concatenate(all_3d_coords, axis=0)
    rgb_values = np.concatenate(all_rgb_values, axis=0)

    return coords, rgb_values


def load_depth_file(filepath: str) -> np.ndarray:
    import OpenEXR

    (exr_part,) = OpenEXR.File(str(filepath)).parts
    return exr_part.channels["V"].pixels.reshape(-1, 1)


def load_rgba_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    rgba = np.array(Image.open(filepath)).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return rgb.reshape(-1, 3), alpha.reshape(-1, 1)


def create_pointcloud_from_multiview(
    multiview_rgb_path: Path,
    multiview_alpha_path: Optional[Path] = None,
    num_points: int = 50000,
    random_subsample_count: int = 2**18,
) -> np.array:

    point_coords_1, point_rgb = merge_multiviews(multiview_rgb_path)

    if multiview_alpha_path is not None:
        point_coords_2, _ = merge_multiviews(multiview_alpha_path, use_color=True)
        point_coords = np.concatenate([point_coords_1, point_coords_2], axis=0)
    else:
        point_coords = point_coords_1

    point_coords_f, _ = subsample_pointcloud(
        point_coords=point_coords,
        num_points=num_points,
        random_subsample_count=random_subsample_count,
    )

    point_rgb_f = colorize_pointcloud(
        point_coords_f, np.concat([point_coords_1, point_rgb], axis=-1)
    )
    return np.concat([point_coords_f, point_rgb_f], axis=-1)
