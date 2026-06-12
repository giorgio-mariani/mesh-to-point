from pathlib import Path
import random
import tempfile
from typing import *
import numpy as np
import tqdm


def visualize_pointcloud(pointcloud: np.ndarray):
    import plotly.graph_objects as go

    assert pointcloud.shape[-1] == 6
    assert len(pointcloud.shape) == 2
    xyz = pointcloud[:, :3]
    x, y, z = xyz.T

    color = [f"rgb({r},{g},{b})" for (r, g, b) in np.uint8(pointcloud[:, 3:] * 255.0)]

    fig = go.Figure(
        dict(
            type="scatter3d",
            mode="markers",
            x=x,
            y=y,
            z=z,
            marker=dict(size=1, color=color),
        )
    )

    fig.update_scenes(aspectmode="data")
    return fig


def create_pointcloud_from_mesh(
    mesh_path: Path,
    num_points: int,
    random_subsample_count: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    from qa3d.pointcloud.render_mesh import render_multiview_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        render_multiview_dataset(mesh_path=mesh_path, output_path=tmpdir, **kwargs)
        pointcloud = create_pointcloud_from_multiview(
            multiview_path=Path(tmpdir),
            num_points=num_points,
            random_subsample_count=random_subsample_count,
        )
    return pointcloud


def create_pointcloud_from_multiview(
    multiview_diffuse_path: Path,
    multiview_alpha_path: Optional[Path] = None,
    num_points: int = 50000,
    random_subsample_count: int = 2**18,
) -> np.array:

    from qa3d.pointcloud.multiview import create_pointcloud_from_multiview as create_pc
    from qa3d.pointcloud.utils import colorize_pointcloud, subsample_pointcloud

    point_coords_1, point_rgb = create_pc(multiview_diffuse_path, use_color=True)

    if multiview_alpha_path is not None:
        point_coords_2, _ = create_pc(multiview_alpha_path, use_color=True)
        point_coords = np.concatenate([point_coords_1, point_coords_2], axis=0)
    else:
        point_coords = point_coords_1

    point_coords_f, _ = subsample_pointcloud(
        point_coords=point_coords,
        num_points=num_points,
        random_subsample_count=random_subsample_count,
    )

    point_rgb_f = colorize_pointcloud(point_coords_f, np.concat([point_coords_1, point_rgb], axis=-1))
    return np.concat([point_coords_f, point_rgb_f], axis=-1)


def get_nearest_neightbor(neighbors, points, batch_size: int = 512):
    num_points, _ = points.shape
    num_batches = num_points // batch_size + (num_points % batch_size != 0)
    nearest_neighbour = np.zeros([num_points], dtype=np.int32)

    for bi in tqdm.tqdm(range(num_batches)):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, num_points)
        batch_points = points[start:end]

        dists = np.sum((neighbors[None, :, :] - batch_points[:, None, :]) ** 2, axis=-1)
        # dists = np.linalg.norm(neighbors[None, :, :] - batch_points[:, None, :], axis=-1)
        nearest_neighbour[start:end] = np.argmin(dists, axis=1)

    return nearest_neighbour


def subsample_pointcloud(
    point_coords: np.ndarray,
    num_points: int,
    point_rgb: Optional[np.ndarray] = None,
    random_subsample_count: Optional[int] = None,
) -> np.ndarray:

    if point_rgb is None:
        point_rgb = np.ones_like(point_coords)

    assert point_coords.shape == point_rgb.shape

    # Downsample points from views
    num_og_points, _ = point_coords.shape
    if random_subsample_count is not None:
        rand_ds_idx = np.random.choice(num_og_points, size=min(num_og_points, random_subsample_count))
        rand_coords = point_coords[rand_ds_idx]
        fps_ds_idx = farthest_point_sample(pointcloud=rand_coords, num_points=num_points)
        fps_coords = rand_coords[fps_ds_idx]
        fps_rgb = point_rgb[rand_ds_idx][fps_ds_idx]
    else:
        fps_ds_idx = farthest_point_sample(pointcloud=point_coords, num_points=num_points)
        fps_coords = point_coords[fps_ds_idx]
        fps_rgb = point_rgb[fps_ds_idx]

    return fps_coords, fps_rgb


def colorize_pointcloud(
    point_coords: np.ndarray,
    point_coords_rgb: np.ndarray,
):
    num_points, _ = point_coords.shape
    num_points_rgb, _ = point_coords_rgb.shape
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors().fit(point_coords)
    nn_inverse = NearestNeighbors().fit(point_coords_rgb[:, :3])

    region_rgb_values = [[] for ci in range(num_points)]
    median_rgb_values = np.zeros([num_points, 3])
    nearest_neighbours = nn.kneighbors(point_coords_rgb[:, :3], 1, return_distance=False).reshape(num_points_rgb)
    nearest_neighbours_2 = nn_inverse.kneighbors(point_coords, 6, return_distance=False).reshape(num_points, 6)

    for vi, ci in enumerate(nearest_neighbours):
        region_rgb_values[ci].append(point_coords_rgb[vi, 3:])

    for ci in range(num_points):
        if len(region_rgb_values[ci]) > 0:
            median_rgb_values[ci, :] = np.mean(region_rgb_values[ci], axis=0)
        else:
            median_rgb_values[ci, :] = np.mean(point_coords_rgb[nearest_neighbours_2[ci], 3:], axis=0)

    return median_rgb_values


def farthest_point_sample(
    pointcloud: np.ndarray,
    num_points: int,
    init_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a subset of the point cloud that is evenly distributed in space.

    First, a random point is selected. Then each successive point is chosen
    such that it is furthest from the currently selected points.

    The time complexity of this operation is O(NM), where N is the original
    number of points and M is the reduced number. Therefore, performance
    can be improved by randomly subsampling points with random_sample()
    before running farthest_point_sample().
    """

    def compute_dists(idx: int) -> np.ndarray:
        # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
        return sq_norms + sq_norms[idx] - 2 * (coords @ coords[idx])

    num_og_points, _ = pointcloud.shape
    coords = pointcloud[:, :3]

    # if num_og_points <= num_points:
    #    return pointcloud

    init_idx = random.randrange(num_og_points) if init_idx is None else init_idx

    indices = np.zeros([num_points], dtype=np.int64)
    indices[0] = init_idx
    sq_norms = np.sum(coords**2, axis=-1)

    cur_distances = compute_dists(init_idx)
    for i in range(1, num_points):
        cur_idx = np.argmax(cur_distances)
        indices[i] = cur_idx

        # Without this line, we may duplicate an index more than once if
        # there are duplicate points, due to rounding errors.
        cur_distances[cur_idx] = -1.0
        cur_distances = np.minimum(cur_distances, compute_dists(cur_idx))

    return indices


def create_camera_poses(
    xy_angles: List[float],
    z_angles: List[float],
    height: int,
    width: int,
    output_file: Union[str, Path],
    force_alpha: Optional[float] = None,
    xy_delta: float = 0.0,
    z_delta: float = 0.0,
) -> List[Dict]:

    import bpy
    import math
    from mathutils import Vector
    import json
    from itertools import product as iproduct

    def set_camera(direction, camera_dist=2.0):
        camera_pos = camera_dist * direction
        bpy.context.scene.camera.location = camera_pos

        rot_quat = direction.to_track_quat("Z", "Y")
        bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()
        bpy.context.view_layer.update()

    def pan_camera(angle, camera_dist=2.0, elevation=0.0):
        direction = [-math.cos(angle), -math.sin(angle), elevation]
        direction = Vector(direction).normalized()
        set_camera(direction, camera_dist=camera_dist)

    def write_camera_parameters(index):
        matrix = bpy.context.scene.camera.matrix_world
        return dict(
            file_path=f"{index:05}_rgba.png",
            depth_file_path=f"{index:05}_depth.exr",
            transform_matrix=[list(c) for c in matrix],
        )

    camera_dist = 2.0
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    fov = bpy.context.scene.camera.data.angle
    # y_fov = bpy.context.scene.camera.data.angle_y

    data = {
        "camera_model": "PINHOLE",
        "fl_x": width / (2 * math.tan(fov / 2)),
        "fl_y": height / (2 * math.tan(fov / 2)),
        "cx": width / 2,
        "cy": height / 2,
        "h": height,
        "w": width,
        "frames": list(),
    }
    if force_alpha is not None:
        data["force_alpha"] = True
        data["force_alpha_value"] = force_alpha

    for i, (xy_angle, z_angle) in enumerate(iproduct(xy_angles, z_angles)):
        xy_noise, z_noise = np.random.random(2) * 2 - 1
        pan_camera(
            angle=xy_angle + xy_noise * xy_delta,
            camera_dist=camera_dist,
            elevation=math.tan(z_angle + z_noise * z_delta),
        )
        data["frames"].append(write_camera_parameters(i))

    with open(output_file, "w") as fp:
        json.dump(data, fp, indent=2)


def optimize_glb(src_glb_path: str, tgt_glb_path: str):
    import bpy

    # Set up empty scene
    bpy.ops.wm.read_homefile(use_empty=True)

    # Import model
    bpy.ops.import_scene.gltf(filepath=src_glb_path)

    # Export model
    bpy.ops.export_scene.gltf(
        filepath=tgt_glb_path,
        export_format="GLB",
        export_apply=True,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="WEBP",
        export_image_quality=75,
        export_keep_originals=False,
        export_texcoords=True,
        export_normals=True,
    )
