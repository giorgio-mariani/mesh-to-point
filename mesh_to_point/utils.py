from pathlib import Path
from typing import *
import numpy as np


def create_camera_poses(
    xy_angles: List[float],
    z_angles: List[float],
    height: int,
    width: int,
    output_file: Union[str, Path],
    force_alpha: Optional[float] = None,
    xy_delta: float = 0.0,
    z_delta: float = 0.0,
) -> List[Dict]:

    import bpy
    import math
    from mathutils import Vector
    import json
    from itertools import product as iproduct

    def set_camera(direction, camera_dist=2.0):
        camera_pos = camera_dist * direction
        bpy.context.scene.camera.location = camera_pos

        rot_quat = direction.to_track_quat("Z", "Y")
        bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()
        bpy.context.view_layer.update()

    def pan_camera(angle, camera_dist=2.0, elevation=0.0):
        direction = [-math.cos(angle), -math.sin(angle), elevation]
        direction = Vector(direction).normalized()
        set_camera(direction, camera_dist=camera_dist)

    def write_camera_parameters(index):
        matrix = bpy.context.scene.camera.matrix_world
        return dict(
            file_path=f"{index:05}_rgba.png",
            depth_file_path=f"{index:05}_depth.exr",
            transform_matrix=[list(c) for c in matrix],
        )

    camera_dist = 2.0
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    fov = bpy.context.scene.camera.data.angle
    # y_fov = bpy.context.scene.camera.data.angle_y

    data = {
        "camera_model": "PINHOLE",
        "fl_x": width / (2 * math.tan(fov / 2)),
        "fl_y": height / (2 * math.tan(fov / 2)),
        "cx": width / 2,
        "cy": height / 2,
        "h": height,
        "w": width,
        "frames": list(),
    }
    if force_alpha is not None:
        data["force_alpha"] = True
        data["force_alpha_value"] = force_alpha

    for i, (xy_angle, z_angle) in enumerate(iproduct(xy_angles, z_angles)):
        xy_noise, z_noise = np.random.random(2) * 2 - 1
        pan_camera(
            angle=xy_angle + xy_noise * xy_delta,
            camera_dist=camera_dist,
            elevation=math.tan(z_angle + z_noise * z_delta),
        )
        data["frames"].append(write_camera_parameters(i))

    with open(output_file, "w") as fp:
        json.dump(data, fp, indent=2)


def optimize_glb(src_glb_path: str, tgt_glb_path: str):
    import bpy

    # Set up empty scene
    bpy.ops.wm.read_homefile(use_empty=True)

    # Import model
    bpy.ops.import_scene.gltf(filepath=src_glb_path)

    # Export model
    bpy.ops.export_scene.gltf(
        filepath=tgt_glb_path,
        export_format="GLB",
        export_apply=True,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="WEBP",
        export_image_quality=75,
        export_keep_originals=False,
        export_texcoords=True,
        export_normals=True,
    )
