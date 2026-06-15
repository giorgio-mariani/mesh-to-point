import tempfile
from pathlib import Path

import bpy

from mesh_to_point.render.scene import prepare_scene, update_camera
from mesh_to_point.render.compositor import setup_compositor
from mesh_to_point.render.config import GlobalConfig, ViewConfig


def render_view(view: ViewConfig, tmp_dir: Path, out_dir: Path):
    # Update camera transform
    update_camera(bpy.context.scene.camera, view.camera)

    # Launch rendering operation
    bpy.ops.render.render(write_still=True)

    # Move outputs to respective filenames
    if view.rgba:
        (tmp_dir / "rgba.png").rename(out_dir / view.rgba)
    if view.mask:
        (alpha,) = (tmp_dir / "alpha").glob("*.png")
        alpha.rename(out_dir / view.mask)
    if view.depth:
        (depth,) = (tmp_dir / "depth").glob("*.exr")
        depth.rename(out_dir / view.depth)
    if view.normals:
        (norm,) = (tmp_dir / "normal").glob("*.exr")
        norm.rename(out_dir / view.normals)


def render_dataset(cfg: GlobalConfig, views: list[ViewConfig]):
    # Prepare scene objects
    prepare_scene(cfg)
        
    with tempfile.TemporaryDirectory() as tmp:
        render_dir = Path(tmp)
    
        # Setup rendering parameters
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.samples = cfg.samples
        bpy.context.scene.use_nodes = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.filepath = str(render_dir / "rgba.png")

        if cfg.use_gpu:
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.use_denoising = True

        for view in views:
            setup_compositor(view, render_dir)
            render_view(view, render_dir, cfg.output_dir)