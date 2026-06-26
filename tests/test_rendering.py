from pathlib import Path
import tempfile
from typing import *
import traceback

import numpy as np
import pytest

from mesh_to_point.camera import CameraModel, CameraPose
from mesh_to_point.io.data import load_rgba_file


@pytest.fixture
def singleview_camera_config() -> Tuple[CameraModel, CameraPose]:
    camera_model = CameraModel(
        camera_id=0,
        model="PINHOLE",
        width=64,
        height=64,
        fx=88.88888249550146,
        fy=88.88888249550146,
        cx=32.0,
        cy=32.0,
    )

    # camera is positioned at [0,-2,0] in world space
    # and points towards the origin
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    t = np.array([0.0, -2.0, 0.0])
    pose = CameraPose(
        image_id=0,
        camera_id=0,
        R=R,
        t=t,
    )

    return camera_model, pose


def compare_histograms(img1: np.ndarray, img2: np.ndarray, bins: int = 128):
    """Return a similarity score between two images.

    The similarity score is based on the correlation between the RGB histograms
    of the two images.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images as HxWxC float32 arrays with values in [0,1.0].
    bins : int
        Number of bins per channel.

    Returns
    -------
    float
        Correlation coefficient between the flattened histograms.
        1.0 means identical, 0.0 means uncorrelated.
    """
    # Compute per‑channel histograms
    hist1 = [np.histogram(img1[..., c], bins=bins, range=(0, 1.0))[0] for c in range(3)]
    hist2 = [np.histogram(img2[..., c], bins=bins, range=(0, 1.0))[0] for c in range(3)]

    # Flatten and normalize
    hist1 = np.concatenate(hist1, axis=0)  # shape (num. bins * 3,)
    hist2 = np.concatenate(hist2, axis=0)  # shape (num. bins * 3, )
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Compute correlation
    mean1 = hist1.mean()
    mean2 = hist2.mean()
    num = np.sum((hist1 - mean1) * (hist2 - mean2))
    den = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))
    return 0.0 if den == 0 else num / den


def test_singleview_rendering(singleview_camera_config):
    from mesh_to_point.render.config import GlobalConfig
    from mesh_to_point.render import render_dataset
    import mesh_to_point.lights as lights

    with tempfile.TemporaryDirectory() as render_dir:
        render_dir = Path(render_dir)
        cam_model, cam_pose = singleview_camera_config

        # Minimal configuration for the compositor test
        bg_light = lights.BackgroundLight()
        bg_light.color = (1.0, 1.0, 1.0, 1.0)
        bg_light.intensity = 0.25
        cfg = GlobalConfig(
            mesh=Path("assets/suzanne.glb"),
            output_dir=render_dir,
            lights=[lights.Light(origin=(0, -0.5, 1), color=(1, 1, 1), intensity=5.0)],
            background_light=bg_light,
            camera=cam_model,
            camera_poses=[cam_pose],
        )

        try:
            render_dataset(cfg)

            # Load the rendered and the reference images
            rendered_img, _ = load_rgba_file(render_dir / "images/0000_rgba.png")
            reference_img, _ = load_rgba_file("tests/assets/reference.png")

            similarity = compare_histograms(rendered_img, reference_img)
            assert similarity > 0.95, f"Histogram similarity too low: {similarity:.4f}"

        except Exception as e:
            traceback.print_exc()  # prints the full stack trace
            tb_str = traceback.format_exc()
            assert False, f"Rendering failed with exception: {tb_str}"
