mesh-to-point — Blender rendering scripts

Contents:
- render_mesh.py  — Blender Python script to import a mesh and render an image.
- run_render.sh   — small wrapper to call Blender with render_mesh.py.

Usage:
1) Make sure Blender is installed and on PATH.
2) Make the wrapper executable (optional): chmod +x run_render.sh
3) Single render example:
   ./run_render.sh /path/to/model.obj /path/to/out.png 1024
   or
   blender --background --python render_mesh.py -- --input model.obj --output out.png --res 1024

Notes:
- Run the Python script using Blender's bundled Python (call via blender --python). Running with system Python will fail because it requires bpy.
- Supported mesh formats: .obj, .glb/.gltf, .ply (Blender importers must be available).
- Default render engine is BLENDER_EEVEE; change to CYCLES with --engine CYCLES if desired.

Batching:
Use a loop to render many files, e.g.:
for f in models/*.obj; do
  out="renders/$(basename "${f%.*}").png"
  ./run_render.sh "$f" "$out" 1024
done

License: MIT
