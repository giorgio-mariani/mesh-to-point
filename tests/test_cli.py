import json
import shutil
import sys
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import pytest

from mesh_to_point.main import main

ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture
def input_mesh():
    return ROOT_DIR / "assets/suzanne.glb"


@pytest.fixture
def input_camera():
    return ROOT_DIR / "tests/assets/single_view.json"


@pytest.fixture
def input_lights():
    return ROOT_DIR / "assets/lights.json"


@pytest.fixture
def pointcloud():
    return np.load(ROOT_DIR / "tests/assets/reference_output/pointcloud.npy")


# ------------------------------------------------------------------
# CLI integration tests (patch heavy functions)
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "flags,expected_calls",
    [
        ([], ["render_dataset", "create_pointcloud_from_multiview"]),
        (["--no-pointcloud"], ["render_dataset"]),
        (
            ["--as-ply"],
            ["render_dataset", "create_pointcloud_from_multiview", "write_ply"],
        ),
        (
            ["--html"],
            [
                "render_dataset",
                "create_pointcloud_from_multiview",
                "create_pointcloud_figure",
            ],
        ),
    ],
)
def test_cli_calls(
    input_mesh, input_camera, input_lights, flags, expected_calls, pointcloud
):

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        def copy_render(*args, **kwargs):
            (tmp_dir / "output").mkdir(parents=True)
            shutil.copytree(
                "tests/assets/reference_output/images", tmp_dir / "output/images"
            )
            shutil.copy(
                "tests/assets/reference_output/transforms.json", tmp_dir / "output"
            )

        # Patch the heavy functions
        with (
            mock.patch(
                "mesh_to_point.main.render_dataset", side_effect=copy_render
            ) as mock_render,
            mock.patch(
                "mesh_to_point.main.create_pointcloud_from_multiview",
                return_value=pointcloud,
            ) as mock_pc,
            mock.patch("mesh_to_point.main.write_ply") as mock_write_ply,
            mock.patch("mesh_to_point.main.create_pointcloud_figure") as mock_fig,
        ):

            # Build the CLI arguments
            args = [
                "--mesh",
                str(input_mesh),
                "--cameras",
                str(input_camera),
                "--lights",
                str(input_lights),
                "--output-dir",
                f"{tmp_dir}/output",
            ] + flags

            # Run the CLI
            with mock.patch.object(sys, "argv", ["mesh-to-point"] + args):
                main()

            # Verify calls
            assert mock_render.called  # <- not working, why?
            if "create_pointcloud_from_multiview" in expected_calls:
                assert mock_pc.called
            else:
                assert not mock_pc.called

            if "write_ply" in expected_calls:
                assert mock_write_ply.called
            else:
                assert not mock_write_ply.called

            if "create_pointcloud_figure" in expected_calls:
                assert mock_fig.called
            else:
                assert not mock_fig.called


@pytest.mark.parametrize("pc_format", ["npy", "ply"])
def test_output_files(input_mesh, input_camera, input_lights, pc_format):
    # Patch heavy functions to avoid real rendering
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path) / "output"
        args = [
            "--mesh",
            str(input_mesh),
            "--cameras",
            str(input_camera),
            "--lights",
            str(input_lights),
            "--output-dir",
            str(tmp_path),
            "--html",
        ] + (["--as-ply"] if pc_format == "ply" else [])

        with mock.patch.object(sys, "argv", ["prog"] + args):
            main()

        # Check that the output directory exists
        assert tmp_path.is_dir()

        # Check that the pointcloud files exist
        assert (tmp_path / f"pointcloud.{pc_format}").exists()
        assert (tmp_path / "pointcloud.html").exists()


# Error handling
def test_missing_required_argument():
    with pytest.raises(SystemExit):
        with mock.patch.object(sys, "argv", ["mesh-to-point"]):
            main()


def test_invalid_mesh_path(tmp_path):
    with pytest.raises(SystemExit):
        with mock.patch.object(
            sys,
            "argv",
            ["mesh-to-point", "--mesh", str(tmp_path / "does_not_exist.obj")],
        ):
            main()
