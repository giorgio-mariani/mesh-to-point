import random
from typing import *

import numpy as np


def get_nearest_neightbor(
    neighbors: np.ndarray, points: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """Return the index of the nearest neighbor for each query point.

    Parameters
    ----------
    neighbors : np.ndarray
        Array of shape ``(N, D)`` containing the candidate points that
        will be searched for. ``D`` is the dimensionality of the
        space (typically 3 for 3D coordinates).
    points : np.ndarray
        Array of shape ``(M, D)`` containing the query points for
        which the nearest neighbor in ``neighbors`` should be found.
    batch_size : int, optional
        Number of query points processed in a single batch.  The
        implementation splits the queries into batches to keep the
        memory footprint low for very large point clouds.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(M,)`` where each element is the
        index of the closest point in ``neighbors`` to the
        corresponding query point.

    Notes
    -----
      If two or more neighbors are at the same distance, the
      first one encountered (i.e., the one with the smallest index)
      is returned.
    """
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
    point_rgb: np.ndarray | None = None,
    random_subsample_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample a point cloud to a target number of points using farthest-point sampling.

    The function applies farthest-point sampling (FPS) to select a representative
    subset of points that are well-distributed across the point cloud. An optional
    random downsampling step can be applied first to reduce the search space for
    very large inputs.

    Parameters
    ----------
    point_coords : np.ndarray
        Array of shape ``(N, D)`` containing the point coordinates, where
        ``N`` is the number of points and ``D`` is the dimensionality.
    num_points : int
        Target number of points to subsample to.
    point_rgb : np.ndarray, optional
        Array of shape ``(N, D)`` containing per-point RGB color values.
        If ``None``, colors default to ones.
    random_subsample_count : int, optional
        If provided, first randomly downsample the point cloud to this
        many points before applying farthest-point sampling. This speeds
        up FPS on very large point clouds at the cost of representativeness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of ``(subsampled_coords, subsampled_rgb)`` arrays, each
        of shape ``(num_points, D)``.

    Notes
    -----
    Farthest-point sampling greedily selects points that are maximally
    distant from already-selected points, producing a uniform spatial
    coverage of the point cloud.
    """
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


def transfer_colors(
    point_coords: np.ndarray,
    point_coords_rgb: np.ndarray,
) -> np.ndarray:
    """Transfer RGB colors from a colored point cloud to an uncolored one.

    For each point in ``point_coords``, the function finds the nearest
    neighbor in ``point_coords_rgb`` and transfers its color.  If multiple
    colored points map to the same target point, their colors are averaged.
    Points with no direct mapping receive the mean color of their 6
    nearest colored neighbors as a fallback.

    Parameters
    ----------
    point_coords : np.ndarray
        Array of shape ``(N, D)`` containing the target point coordinates
        that need to be colored.
    point_coords_rgb : np.ndarray
        Array of shape ``(M, D + 3)`` containing source point coordinates
        followed by their RGB color values (the last 3 columns).

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` containing the mean RGB color for each
        point in ``point_coords``.
    """
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
    init_idx: int | None = None,
) -> np.ndarray:
    """Select a subset of points that are evenly spread in space.

    The algorithm is a classic *farthest-point sampling* (FPS).  It
    starts from an arbitrary point (random if ``init_idx`` is ``None``)
    and repeatedly adds the point that is farthest from the set of
    already selected points.  The result is a set of ``num_points``
    indices that tend to cover the whole point cloud.

    Parameters
    ----------
    pointcloud : np.ndarray
            Array of shape ``(N, D)`` where ``N`` is the number of
            points and ``D`` is the dimensionality (usually 3 for 3D
            coordinates).  Only the first three columns are used for the
            distance calculation.
    num_points : int
            Desired number of points to sample.  If ``num_points`` is
            larger than ``N`` the function will simply return the
            original indices.
    init_idx : int | None, optional
            Index of the point to start from.  If ``None`` a random
            point is chosen.

    Returns
    -------
    np.ndarray
            Integer array of shape ``(num_points,)`` containing the
            indices of the selected points.

    Notes
    -----
    * The algorithm runs in `O(NM)` time where ``N`` is the
        number of input points and ``M`` is ``num_points``.
    * For very large point clouds it is advised to first reduce the
        dataset with a random subsample before calling FPS.
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


def create_pointcloud_figure(pointcloud: np.ndarray) -> "plotly.graph_objects.Figure":
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
