from pathlib import Path

import pytest
import numpy as np

from mesh_to_point.pointcloud.misc import (
    subsample_pointcloud,
    farthest_point_sample,
)
from mesh_to_point.pointcloud.multiview import (
    merge_multiviews,
    create_pointcloud_from_multiview,
)


def chamfer_l2_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:

    from sklearn.neighbors import NearestNeighbors

    # Fit nearest neighbors on point clouds
    nbrs_X = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(X)
    nbrs_Y = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Y)

    # Get distances from X to Y
    distances_X, _ = nbrs_Y.kneighbors(X)
    # Get distances from Y to X
    distances_Y, _ = nbrs_X.kneighbors(Y)

    score_X = np.sqrt(np.mean(distances_X**2))
    score_Y = np.sqrt(np.mean(distances_Y**2))

    return score_X + score_Y


@pytest.fixture(scope="module")
def reference_multiview_dir():
    """Path to the reference multiview output directory."""
    return Path(__file__).parent / "assets" / "reference_output"


@pytest.fixture(scope="module")
def reference_pointcloud():
    """Load the reference pointcloud from disk."""
    return np.load(
        Path(__file__).parent / "assets" / "reference_output" / "pointcloud.npy"
    )


def test_farthest_point_sample():
    # Create a simple 2D point cloud
    pc = np.array([[0, 0], [-1, 0], [0.5, 1], [1.5, 1.5], [0.5, 0.5]])

    # Sample 3 points
    indices = farthest_point_sample(pc, 3, init_idx=0)

    # Should have 3 unique indices
    assert len(set(indices)) == 3
    # The first index should be 0
    assert indices[0] == 0
    assert np.all(indices == [0, 3, 2]), indices


@pytest.mark.parametrize("subsample_size", [5, 10, 20])
def test_subsample_pointcloud(subsample_size):
    pc_coords = np.random.rand(100, 3)
    pc_rgb = np.zeros([100, 3])

    subsampled, subsampled_rgb = subsample_pointcloud(
        point_coords=pc_coords,
        num_points=subsample_size,
        point_rgb=pc_rgb,
    )
    assert subsampled.shape == (subsample_size, 3)
    assert subsampled_rgb.shape == (subsample_size, 3)

    # Ensure points are from original set
    for point_coord, point_rgb in zip(subsampled, subsampled_rgb):
        distances = np.linalg.norm(point_coord - subsampled, axis=1)
        assert np.any(np.isclose(distances, 0.0))

        idx = np.argmax(np.isclose(distances, 0.0))
        # assert False, (np.isclose(distances, 0.0).shape, idx.shape)
        assert np.isclose(np.linalg.norm(point_rgb - pc_rgb[idx]), 0.0)


def test_returns_coords_and_rgb(reference_multiview_dir, reference_pointcloud):
    coords, rgb = merge_multiviews(reference_multiview_dir)
    assert isinstance(coords, np.ndarray)
    assert isinstance(rgb, np.ndarray)

    assert coords.ndim == 2
    assert coords.shape[1] == 3  # (x, y, z)

    assert rgb.ndim == 2
    assert rgb.shape[1] == 3  # (r, g, b)

    assert coords.shape[0] == rgb.shape[0]

    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)

    assert not np.any(np.isnan(coords))
    ref_coords = reference_pointcloud[:, :3]

    assert chamfer_l2_distance(coords, ref_coords) <= 0.01


@pytest.mark.parametrize("num_points", [10, 1000, 5000])
@pytest.mark.parametrize("subsample_count", [10, 5000, 2**18])
def test_returns_correct_shape(num_points, subsample_count, reference_multiview_dir):
    pc = create_pointcloud_from_multiview(
        multiview_rgb_path=reference_multiview_dir,
        num_points=num_points,
        random_subsample_count=subsample_count,
    )

    assert pc.ndim == 2
    assert pc.shape[1] == 6  # (x, y, z, r, g, b)
    assert pc.shape[0] == num_points

    coords, rgb = pc[:, :3], pc[:, 3:]

    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)
    assert not np.any(np.isnan(pc))


def test_matches_reference_coords(reference_multiview_dir, reference_pointcloud):
    """Generated pointcloud coordinates should be close to the reference."""
    pc = create_pointcloud_from_multiview(
        multiview_rgb_path=reference_multiview_dir,
        num_points=50000,
        random_subsample_count=2**18,
    )

    coords, rgb = pc[:, :3], pc[:, 3:]
    ref_coords = reference_pointcloud[:, :3]
    assert chamfer_l2_distance(coords, ref_coords) <= 0.01

    gen_rgb_mean = pc[:, 3:].mean()
    ref_rgb_mean = reference_pointcloud[:, 3:].mean()
    assert np.isclose(gen_rgb_mean, ref_rgb_mean, rtol=0.001)
