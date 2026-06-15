import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import *

import bpy
from mathutils import Vector
import numpy as np


@dataclass
class CameraConfig:
    origin: tuple = (0, 0, 0)
    x_fov: float = 0.0
    x: tuple = (0, 0, 0)
    y: tuple = (0, 0, 0)
    z: tuple = (0, 0, 0)


@dataclass
class SingleViewConfig:
    camera: CameraConfig
    resolution_x: int
    resolution_y: int
    depth_file_path: Optional[Path] = None
    rgba_file_path: Optional[Path] = None
    mask_file_path: Optional[Path] = None
    normals_file_path: Optional[Path] = None

    @property
    def depth_pass(self) -> bool:
        return self.depth_file_path is not None

    @property
    def rgba_pass(self) -> bool:
        return self.rgba_file_path is not None

    @property
    def mask_pass(self) -> bool:
        return self.mask_file_path is not None

    @property
    def normal_pass(self) -> bool:
        return self.normals_file_path is not None


@dataclass
class LightConfig:
    origin: tuple
    color: tuple
    intensity: float
    use_shadows: bool = True


@dataclass
class MultiViewRenderingConfig:
    camera_poses: List[SingleViewConfig]
    lights: List[LightConfig]
    mesh: Optional[Path] = None
    background_color: tuple = (1.0, 1.0, 1.0, 1.0)
    background_strength: float = 1.0
    force_alpha: bool = False
    force_alpha_value: float = 1.0
    use_gpu: bool = False


def create_camera():
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    return camera_object


def create_light(light: LightConfig):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = light.intensity
    light_data.angle = 0.5 * math.pi / 180
    light_data.use_shadow = light.use_shadows
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -light.origin
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = light.origin


def create_background_environment(color: tuple, strength: float):
    shader_tree = bpy.context.scene.world.node_tree
    background_node = shader_tree.nodes["Background"]
    background_node.inputs["Color"].default_value = color
    background_node.inputs["Strength"].default_value = strength


def update_camera(camera, cam_pose: CameraConfig):
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = cam_pose.x_fov

    camera.location[:] = cam_pose.origin
    camera.matrix_world.col[0][:3] = cam_pose.x
    camera.matrix_world.col[1][:3] = np.array(cam_pose.y)  # * -1
    camera.matrix_world.col[2][:3] = np.array(cam_pose.z)  # * -1
    camera.matrix_world.col[3][:3] = cam_pose.origin
    bpy.context.view_layer.update()


def scene_bbox(single_obj=None, ignore_matrix=False):
    def scene_meshes():
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj

    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    def scene_root_objects():
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    # Apply transform (enforce common origin, which is necessary for proper transforms)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def override_alpha_val(alpha: float):
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):

            # Remove existing materials and add a new one
            # obj.data.materials.clear()
            # mat = bpy.data.materials.new(name="VertexColored")
            # obj.data.materials.append(mat)

            for mat in obj.data.materials:
                mat.use_nodes = True

                # There should be a Principled BSDF by default.
                bsdf_node = None
                for node in mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        bsdf_node = node
                assert (
                    bsdf_node is not None
                ), "material has no Principled BSDF node to modify"

                bsdf_node.inputs["Alpha"].default_value = alpha
                # mat_index = len(obj.data.materials)
                # polygon_num = len(obj.data.polygons)
                # for i in range(polygon_num):
                #    obj.data.polygons[i].material_index = mat_index


def create_outputfile_subgraph(
    input_socket,
    compositor_tree,
    output_path: str,
    color_mode: str = "RGB",
    file_format: str = "OPEN_EXR",
    color_depth: str = "16",
):
    tmp_node = compositor_tree.nodes.new(type="CompositorNodeOutputFile")
    tmp_node.format.file_format = file_format
    tmp_node.format.color_mode = color_mode
    tmp_node.format.color_depth = color_depth
    tmp_node.base_path = f"{output_path}"
    compositor_tree.links.new(input_socket, tmp_node.inputs[0])


def setup_scene(render_config: MultiViewRenderingConfig, rendering_dir: str):
    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import model
    bpy.ops.import_scene.gltf(filepath=str(render_config.mesh))
    normalize_scene()

    # Uniform alpha to single value
    if render_config.force_alpha:
        override_alpha_val(render_config.force_alpha_value)

    # Create scene lights
    for light in render_config.lights:
        create_light(light)
    create_background_environment(
        render_config.background_color, render_config.background_strength
    )

    # Create scene camera
    create_camera()

    # Setup rendering config
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = f"{rendering_dir}/rgba.png"

    if render_config.use_gpu:
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.use_denoising = True


def setup_compositor(sv_config: SingleViewConfig, rendering_dir: str):
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = sv_config.depth_pass
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = sv_config.normal_pass
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = sv_config.resolution_x
    bpy.context.scene.render.resolution_y = sv_config.resolution_y

    tree = bpy.context.scene.node_tree

    # Clean up scene
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Setup input node
    input_node = tree.nodes.new(type="CompositorNodeRLayers")

    if sv_config.mask_pass:
        math_node = tree.nodes.new(type="CompositorNodeMath")
        math_node.operation = "GREATER_THAN"
        tree.links.new(input_node.outputs["Alpha"], math_node.inputs[0])
        math_node.inputs[1].default_value = 0.0001
        create_outputfile_subgraph(
            math_node.outputs["Value"],
            tree,
            f"{rendering_dir}/alpha",
            "BW",
            file_format="PNG",
            color_depth="8",
        )

    if sv_config.depth_pass:
        create_outputfile_subgraph(
            input_node.outputs["Depth"], tree, f"{rendering_dir}/depth", "BW"
        )

    if sv_config.normal_pass:
        create_outputfile_subgraph(
            input_node.outputs["Normal"], tree, f"{rendering_dir}/normal"
        )


def render_and_store(
    sv_config: SingleViewConfig,
    rendering_dir: Path,
    output_dir: Path,
):

    update_camera(
        camera=bpy.context.scene.camera,
        cam_pose=sv_config.camera,
    )

    # Run rendering engine
    bpy.ops.render.render(write_still=True)

    # Clean up procedures
    if sv_config.rgba_pass:
        os.rename(rendering_dir / "rgba.png", output_dir / sv_config.rgba_file_path)

    if sv_config.mask_pass:
        (alpha_file,) = (rendering_dir / "alpha").glob("*.png")
        os.rename(alpha_file, output_dir / sv_config.mask_file_path)

    # The output depth image must be moved to target file
    if sv_config.depth_pass:
        (depthmap_file,) = (rendering_dir / "depth").glob("*.exr")
        os.rename(depthmap_file, output_dir / sv_config.depth_file_path)

    # The output normals must be moved to target file
    if sv_config.normal_pass:
        (normal_file,) = (rendering_dir / "normal").glob("*.exr")
        os.rename(normal_file, output_dir / sv_config.normals_file_path)


def render_multiview_dataset(
    output_path: Union[str, Path],
    render_config: MultiViewRenderingConfig,
):

    output_path = Path(output_path)
    if render_config.mesh is None:
        raise ValueError("Input mesh must be specified!")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Prepare scene
        setup_scene(render_config, tmp_dir)
        for sv_config in render_config.camera_poses:
            setup_compositor(sv_config, tmp_dir)
            render_and_store(sv_config, tmp_dir, output_path)


def main():
    import sys
    import json

    try:
        dash_index = sys.argv.index("--")
    except ValueError as exc:
        raise ValueError("arguments must be preceded by '--'") from exc

    raw_args = sys.argv[dash_index + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--cameras-path", type=str)
    parser.add_argument("--output-path", required=False, type=str, default="multiviews")

    args = parser.parse_args(raw_args)

    if not Path(args.input_path).exists():
        raise ValueError(f"input path does not exist: {args.input_path}")

    if not Path(args.camera_pose_path).exists():
        raise ValueError(f"camera path does not exist: {args.camera_pose_path}")
    with open(args.camera_pose_path, "r") as fp:
        camera_data = json.load(fp)

    height = camera_data["h"]
    width = camera_data["w"]
    x_fov = 2 * math.atan(width / 2 / camera_data["fl_x"])

    cam_poses = []
    for p in camera_data["frames"]:
        transform_matrix = np.array(p["transform_matrix"])

        cam_pose = CameraConfig()
        cam_pose.origin = transform_matrix[:3, 3]
        cam_pose.x_fov = x_fov
        cam_pose.x = transform_matrix[:3, 0]
        cam_pose.y = transform_matrix[:3, 1]
        cam_pose.z = transform_matrix[:3, 2]

        sv_config = SingleViewConfig(
            camera=cam_pose,
            depth_file_path=p.get("depth_file_path", None),
            rgba_file_path=p.get("file_path", None),
            normals_file_path=p.get("normal_file_path", None),
            mask_file_path=p.get("mask_path", None),
            resolution_x=width,
            resolution_y=height,
        )

        cam_poses.append(sv_config)

    cfg = MultiViewRenderingConfig(
        camera_poses=cam_poses,
        lights=[],
        mesh=Path(args.input_path),
        background_color=(1.0, 1.0, 1.0, 1.0),
        background_strength=1.0,
        force_alpha=camera_data.get("force_alpha", False),
        force_alpha_value=camera_data.get("force_alpha_value", 1.0),
        use_gpu=True,
    )

    render_multiview_dataset(output_path=args.output_path, render_config=cfg)


if __name__ == "__main__":
    main()
