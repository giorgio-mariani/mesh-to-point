#!/usr/bin/env python3
"""
Blender script to render a mesh to an image.
Run with:
  blender --background --python render_mesh.py -- --input /path/to/mesh.obj --output /path/to/out.png --res 1024
Supports .obj, .glb/.gltf, .ply (if importer enabled).
"""
import sys
import os
import argparse

try:
    import bpy
    import mathutils
except Exception:
    print("This script must be run inside Blender. Example:\n  blender --background --python render_mesh.py -- --input mesh.obj --output out.png")
    sys.exit(1)


def clear_scene():
    # remove all objects and data blocks
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=path)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == '.ply':
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError(f"Unsupported mesh extension: {ext}")
    # return mesh objects in the scene
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']


def frame_objects(objs):
    # compute world-space bounding box min/max
    coords = []
    for o in objs:
        for v in o.bound_box:
            coords.append(o.matrix_world @ mathutils.Vector(v))
    if not coords:
        return (0.0, 0.0, 0.0), 1.0
    xs = [v.x for v in coords]
    ys = [v.y for v in coords]
    zs = [v.z for v in coords]
    minv = (min(xs), min(ys), min(zs))
    maxv = (max(xs), max(ys), max(zs))
    center = ((minv[0]+maxv[0])/2.0, (minv[1]+maxv[1])/2.0, (minv[2]+maxv[2])/2.0)
    dims = (maxv[0]-minv[0], maxv[1]-minv[1], maxv[2]-minv[2])
    max_dim = max(dims) if max(dims) > 0 else 1.0
    return center, max_dim


def setup_camera(target, distance, resolution):
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam)
    # position camera: offset on Y axis and slightly above
    cam.location = (target[0], target[1] - distance * 1.5, target[2] + distance * 0.25)
    # point to target
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    empty = bpy.data.objects.new('CameraTarget', None)
    empty.location = target
    bpy.context.collection.objects.link(empty)
    cam_constraint.target = empty
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    return cam


def setup_light(target, distance):
    light_data = bpy.data.lights.new(name='Sun', type='SUN')
    light = bpy.data.objects.new(name='Sun', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (target[0] + distance, target[1] - distance, target[2] + distance)
    light.data.energy = 5.0
    return light


def ensure_material(objs):
    for o in objs:
        if not o.data.materials:
            mat = bpy.data.materials.new(name='material')
            mat.use_nodes = True
            o.data.materials.append(mat)


def render(output_path, engine='BLENDER_EEVEE', samples=16):
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == 'CYCLES' and hasattr(scene, 'cycles'):
        scene.cycles.samples = samples
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def parse_args():
    # arguments after "--" when running blender
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description='Render mesh to image (Blender)')
    parser.add_argument('--input', '-i', required=True, help='Input mesh file (.obj/.glb/.ply)')
    parser.add_argument('--output', '-o', required=True, help='Output image path (png)')
    parser.add_argument('--res', type=int, default=1024, help='Square resolution (px)')
    parser.add_argument('--engine', default='BLENDER_EEVEE', help='Render engine: BLENDER_EEVEE or CYCLES')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    clear_scene()
    objs = import_mesh(args.input)
    center, max_dim = frame_objects(objs)
    setup_camera(center, max_dim * 1.8, args.res)
    setup_light(center, max_dim * 2.0)
    ensure_material(objs)
    # ensure parent dir exists for output
    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    render(args.output, engine=args.engine)

if __name__ == '__main__':
    main()
