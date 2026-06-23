import math
from enum import Enum
from typing import *

import bpy
from mathutils import Vector

from mesh_to_point.camera import CameraPose
from mesh_to_point.lights import Light

from mesh_to_point.render.config import GlobalConfig


class MeshFormat(Enum):
    """Supported mesh file formats."""

    GLTF = ("gltf", "import_scene", "gltf")
    GLB = ("glb", "import_scene", "gltf")
    OBJ = ("obj", "import_scene", "obj")
    FBX = ("fbx", "import_scene", "fbx")
    PLY = ("ply", "import_mesh", "ply")
    STL = ("stl", "import_mesh", "stl")

    def __init__(self, extension: str, load_module: str, load_operator: str):
        self.extension = extension
        self.load_module = load_module
        self.load_operator = load_operator

    @classmethod
    def from_string(cls, format_str: str):
        """Convert a string (e.g., 'gltf', 'glb') to a MeshFormat enum member."""
        format_str = format_str.lower()
        for member in cls:
            if member.extension == format_str:
                return member
        raise ValueError(
            f"Unsupported format: {format_str}. Supported: {[m.extension for m in cls]}"
        )


def _scene_bounding_box(single_obj=None, ignore_matrix=False) -> Tuple[Vector, Vector]:
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


def _normalize_scene():
    def scene_root_objects():
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    bbox_min, bbox_max = _scene_bounding_box()
    scale = 1 / max(bbox_max - bbox_min)

    # Apply transform (enforce common origin, which is necessary for proper transforms)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = _scene_bounding_box()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def _create_camera():
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    return camera_object


def _create_light(light: Light):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = light.intensity
    light_data.angle = 0.5 * math.pi / 180
    light_data.use_shadow = light.use_shadows
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -Vector(light.origin)
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = light.origin


def _set_background_color(color: tuple, strength: float):
    shader_tree = bpy.context.scene.world.node_tree
    background_node = shader_tree.nodes["Background"]
    background_node.inputs["Color"].default_value = color
    background_node.inputs["Strength"].default_value = strength


def _override_alpha_val(alpha: float):
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


def load_input_mesh(mesh_path: str, format: MeshFormat | str, **kwargs: dict) -> List:
    """Load mesh in the current blender scene.

    Supported mesh formats can be found in :class:`MeshFormat`.

    Parameters
    ----------
    mesh_path:
        Path to the mesh file
    format:
        Mesh format (MeshFormat enum or string like 'gltf', 'obj', etc.)
    kwargs:
        Additional keyword arguments to pass to the importer

    Returns
    -------
        List of imported objects
    """
    # Convert string to enum if needed
    if isinstance(format, str):
        format = MeshFormat.from_string(format)

    # Get the operator function
    importer = getattr(getattr(bpy.ops, format.load_module), format.load_operator)

    # Store the current objects to identify new ones
    objects_before = set(bpy.context.scene.objects)

    # Call the importer
    importer(filepath=str(mesh_path), **kwargs)

    # Return the newly imported objects
    imported_objects = list(set(bpy.context.scene.objects) - objects_before)

    return imported_objects


def update_camera(camera_obj, camera_pose: CameraPose) -> None:
    """Update a Blender camera object to match the provided extrinsic parameters.

    The function applies the input extrinsic parameters to the blender
    scene camera.

    Parameters
    ----------
    camera_obj:
        The blender camera object to update.
    camera_pose: CameraPose
        The camera's extrinsic parameters.

    """

    R = camera_pose.R
    x, y, z = R.T

    # TODO: setup the commented stuff
    # cam_obj.data.sensor_fit = "HORIZONTAL"
    # cam_obj.data.angle = camera_pose.fov
    # cam_obj.location[:] = camera_pose.origin
    camera_obj.matrix_world.col[0][:3] = x
    camera_obj.matrix_world.col[1][:3] = y
    camera_obj.matrix_world.col[2][:3] = z
    camera_obj.matrix_world.col[3][:3] = camera_pose.t
    bpy.context.view_layer.update()


def prepare_scene(cfg: GlobalConfig) -> None:
    """Prepare the Blender scene according to the provided configuration.

    This function performs the following steps:
    1. Clears the current scene of all objects.
    2. Imports the input mesh specified in ``cfg.mesh`` using the GLTF importer.
    3. Normalises the imported geometry so that it fits within a unit cube and
       is centred at the origin.
    4. Optionally overrides the alpha channel of the mesh if ``cfg.force_alpha``
       is set.
    5. Adds lights defined in ``cfg.lights``.
    6. Sets the world background colour and strength from ``cfg.background_*``.
    7. Creates a default camera.

    Parameters
    ----------
    cfg: GlobalConfig
        Configuration object containing mesh path, lighting, background camera
        parameters.
    """
    # Cleanup scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Load input object
    bpy.ops.import_scene.gltf(filepath=str(cfg.mesh))
    _normalize_scene()

    if cfg.force_alpha:
        _override_alpha_val(cfg.force_alpha_value)

    for light in cfg.lights:
        _create_light(light)

    _set_background_color(cfg.background_light.color, cfg.background_light.intensity)
    _create_camera()
