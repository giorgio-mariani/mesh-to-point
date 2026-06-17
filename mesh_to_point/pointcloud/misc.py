import random
from typing import *

import numpy as np


def get_nearest_neightbor(
    neighbors: np.ndarray, points, batch_size: int = 512
) -> np.ndarray:
    num_points, _ = points.shape
    num_batches = num_points // batch_size + (num_points % batch_size != 0)
    nearest_neighbour = np.zeros([num_points], dtype=np.int32)

    for bi in range(num_batches):
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
        rand_ds_idx = np.random.choice(
            num_og_points, size=min(num_og_points, random_subsample_count)
        )
        rand_coords = point_coords[rand_ds_idx]
        fps_ds_idx = farthest_point_sample(
            pointcloud=rand_coords, num_points=num_points
        )
        fps_coords = rand_coords[fps_ds_idx]
        fps_rgb = point_rgb[rand_ds_idx][fps_ds_idx]
    else:
        fps_ds_idx = farthest_point_sample(
            pointcloud=point_coords, num_points=num_points
        )
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
    nearest_neighbours = nn.kneighbors(
        point_coords_rgb[:, :3], 1, return_distance=False
    ).reshape(num_points_rgb)
    nearest_neighbours_2 = nn_inverse.kneighbors(
        point_coords, 6, return_distance=False
    ).reshape(num_points, 6)

    for vi, ci in enumerate(nearest_neighbours):
        region_rgb_values[ci].append(point_coords_rgb[vi, 3:])

    for ci in range(num_points):
        if len(region_rgb_values[ci]) > 0:
            median_rgb_values[ci, :] = np.mean(region_rgb_values[ci], axis=0)
        else:
            median_rgb_values[ci, :] = np.mean(
                point_coords_rgb[nearest_neighbours_2[ci], 3:], axis=0
            )

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
