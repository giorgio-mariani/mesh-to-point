import tempfile
from pathlib import Path

import bpy

from mesh_to_point.render.io import write_cameras_txt, write_images_txt, write_json
from mesh_to_point.render.scene import prepare_scene, update_camera
from mesh_to_point.render.compositor import setup_compositor
from mesh_to_point.render.config import GlobalConfig


def _render_view(
    cfg: GlobalConfig, pose_index: int, render_dir: Path, out_dir: Path
) -> None:
    camera = cfg.camera
    camera_pose = cfg.camera_poses[pose_index]

    # Update camera transform
    update_camera(bpy.context.scene.camera, camera_pose)

    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.resolution_y = camera.height
    scene.render.resolution_x = camera.width

    # Launch rendering operation
    bpy.ops.render.render(write_still=True)

    view_id = "{:04d}".format(camera_pose.image_id)

    # Move outputs to respective filenames
    (render_dir / "rgba.png").rename(out_dir / f"{view_id}_rgba.png")

    if cfg.depth_pass:
        (render_dir / "depth.exr").rename(out_dir / f"{view_id}_depth.exr")


def render_dataset(cfg: GlobalConfig) -> None:
    """Render an entire dataset of multiple views of a single object.

    Parameters
    ----------
    cfg:
        Global configuration object containing camera settings, camera poses,
        rendering options and output directories.

    Behaviour
    ---------
    1. Ensures ``cfg.output_dir`` exists and is empty; otherwise raises a
       :class:`RuntimeError` to prevent accidental overwrites.
    2. Calls :func:`prepare_scene` to populate the Blender scene with the
       geometry and materials defined in the configuration.
    3. Creates a temporary directory for intermediate render files.
    4. Configures Blender's Cycles engine, node system, sample count, and
       output format. If ``cfg.use_gpu`` is ``True`` the GPU device is
       selected and denoising is enabled.
    5. Sets up the compositor via :func:`setup_compositor`.
    6. Iterates over all camera poses, rendering each view.

    The function does not return a value; all output is written to
    ``cfg.output_dir``.
    """

    # Ensure output directory doesn't exists or is empty
    out_dir = cfg.output_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Output directory '{out_dir}' is not empty.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "images").mkdir()
    (out_dir / "sparse").mkdir()

    write_cameras_txt(out_dir / "sparse/cameras.txt", [cfg.camera])
    write_images_txt(out_dir / "sparse/images.txt", cfg.camera_poses)
    write_json(out_dir / "transform.json", cfg.camera, cfg.camera_poses)

    # Prepare scene objects
    prepare_scene(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        render_dir = Path(tmp)

        # Setup rendering parameters
        scene = bpy.context.scene

        scene.view_layers["ViewLayer"].use_pass_z = True
        scene.render.engine = "CYCLES"
        scene.use_nodes = True
        scene.cycles.samples = cfg.samples
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(render_dir / "rgba.png")

        if cfg.use_gpu:
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.use_denoising = True

        setup_compositor(cfg, render_dir)

        for view_index in range(len(cfg.camera_poses)):
            _render_view(cfg, view_index, render_dir, cfg.output_dir / "images")
