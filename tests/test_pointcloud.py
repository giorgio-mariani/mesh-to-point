import pytest
import numpy as np
from mesh_to_point.pointcloud.misc import (
    subsample_pointcloud,
    colorize_pointcloud,
    farthest_point_sample,
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
