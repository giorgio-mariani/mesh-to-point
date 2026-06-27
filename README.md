# Mesh‑to‑Point
![PyPI version](https://img.shields.io/pypi/v/mesh-to-point?style=flat-square)

## Overview
`mesh-to-point` is a lightweight Python library that converts 3D meshes into dense RGB point clouds.

It is built on top of **NumPy**, and **scikit‑learn** for data handling, and uses **Blender** (via the `bpy` API) for rendering synthetic views.


> **Why use it?**
> * Easy installation using the headless Blender `bpy` library.
> * Simple CLI for quick experiments.
> * Clean Python API for integration into pipelines.

### Features

* Render a mesh from arbitrary camera poses.
* Generate depth maps and RGB images.
* Build a dense RGB point cloud from multiple views.
* Export point clouds as NumPy arrays, PLY files, or interactive Plotly HTML.
* Customizable lighting and camera configurations via JSON.

## Installation
The package is published on PyPI and can be installed with `pip`:

```bash
pip install mesh-to-point
```

Note that this will also install a headless version of blender 5.0 to be used for rendering.

**Development install.** The repository is set up to use **uv** for dependency and environment management. If you plan to contribute or run the test suite, follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/giorgio-mariani/mesh-to-point.git
    cd mesh-to-point
    ```
2. **Create a virtual environment and install dev dependencies**
    ```bash
    uv sync --extra dev
    pre-commit install
    ```
    These commands set up a clean environment with all the tools needed for development and testing.

## Quick start
### Command Line Interface
```bash
# Render a mesh and generate a point cloud with 8192 points
mesh-to-point --mesh assets/suzanne.glb \
              --cameras assets/cameras.json \
              --lights assets/lights.json \
              --output-dir output \
              --points 8192 \
              --html
```

The command will:

* Render the mesh from the specified camera viewpoints using Blender Cycles.
* Store the rendered RGB-D images in the chosen output directory.
* Build a dense RGB point cloud from the rendered images.
* Write the point cloud as a NumPy ``.npy`` array or (optionally) as a text ``.ply`` file.
* Optionally generate an interactive Plotly HTML visualization of the point cloud.

#### Output directory
The output directory contains the rendered images, camera transforms, point cloud data, and an optional HTML visualization. Images are stored under ``<output_dir>/images`` as ``<index>_rgb.png`` and ``<index>_depth.exr``. Extrinsic and intrinsic camera parameters are saved in ``transforms.json``. The point cloud is written as ``pointcloud.npy`` (or ``pointcloud.ply`` if ``--as-ply`` is used). If ``--html`` is enabled, an interactive Plotly visualization is generated as ``pointcloud.html``.

Overview of output directory structure:

```
<output-dir>
|- images/
|  |- 0000_depth.exr
|  |- 0000_rgb.png
|  |  ...
|  |- <num-views>_rgb.png
|
|- transforms.json  # Camera parameters
|- pointcloud.npy   # Point cloud data (`pointcloud.ply` if --as-ply is used)
|- pointcloud.html  # Point cloud visualization script (only if --html is used)
```

#### Available input formats
`mesh-to-point` accepts a variety of common 3D mesh file formats. The format is detected automatically from the file extension.

| Extension | Importer used | Notes |
|-----------|---------------|-------|
| `.gltf`   | `bpy.ops.import_scene.gltf` | Standard GLTF 2.0, supports embedded textures.
| `.glb`    | `bpy.ops.import_scene.gltf` | Binary GLTF, same importer as `.gltf`.
| `.obj`    | `bpy.ops.import_scene.obj` | Wavefront OBJ
| `.fbx`    | `bpy.ops.import_scene.fbx` | Autodesk FBX
| `.ply`    | `bpy.ops.import_mesh.ply` | Stanford PLY, supports ASCII/Binary.
| `.stl`    | `bpy.ops.import_mesh.stl` | STL, supports ASCII/Binary.

All imports are performed by Blender’s built‑in import operators. `mesh-to-point` does not attempt any format conversion; the mesh is loaded directly into the Blender scene and then normalised to fit within a unit cube centred at the origin.


#### Configuration files
Lighting and camera information are provided to the CLI through JSON files; These describe the extrinsic camera poses and the intrinsic camera parameters, together with the amount of point lights in the scene and the background light color and intensity. If these are not provided, `mesh-to-point` will default to [[assets/cameras.json]] and [[assets/lights.json]] for cameras and lights respectively.
- *Camera configuration.* The file should follow the format used by the `mesh_to_point.read_camera_config` helper. It contains intrinsic parameters (focal length, principal point, image size) and a list of extrinsic pose matrices.
- *Lights configuration.* It specifies an optional environment color and intensity, together with a list of point lights. Each light has an origin, color, intensity, and a flag for shadows.  The file is parsed by `mesh_to_point.read_light_config`, take a look at the function docstring for a more in-depth description.

> **NOTE ⚠**
When loaded into the scene, the input geometry mesh is automatically normalized to match the size of a unit cube, and placed at the origin. Camera and light parameters should take this into consideration.


Below are two minimal examples; one for a camera configuration file and another for a light configuration file.

*Minimal cameras config file:*
```json
{
    "camera_model": "PINHOLE",  # Only PINHOLE or SIMPLE_PINHOLE are supported
    "fl_x": 1422.22,
    "fl_y": 1422.22,
    "cx": 512,
    "cy": 512,
    "h": 1024,
    "w": 1024,
    "frames": [
        {
            "transform_matrix": [  # Camera-to-world transformation matrix
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        }
    ]
}
```

*Minimal lights config file:*
```json
lights.json
{
    "environment_color": [1, 1, 1, 1],  # RGB-Alpha
    "environment_intensity": 0.5,       # Between  [0.0, 1.0]
    "lights": [
        {
            "origin": [0, 0, 2],
            "color": [1, 1, 1],
            "intensity": 10,            # Between [0.0, +inf)
            "use_shadows": true
        }
    ]
}
```

Examples for both configuration file types can found in the `./assets` directory.

---

## Python API

The library exposes three main functions:

| Function | Purpose |
|----------|---------|
| `render_dataset` | Render RGB‑D images from a mesh. |
| `create_pointcloud_from_multiview` | Build a point cloud from a directory of rendered images. |
| `create_pointcloud_figure` | Create a Plotly figure for visualising a point cloud. |

### Example workflow

```python
from mesh_to_point import (
        GlobalConfig,
        read_camera_config,
        read_light_config,
        render_dataset,
        create_pointcloud_from_multiview,
        create_pointcloud_figure,
)

# Load configuration
camera, camera_poses = read_camera_config("assets/cameras.json")
env_light, lights = read_light_config("assets/lights.json")

# Rendering configuration
cfg = GlobalConfig(
        mesh="assets/suzanne.glb",
        output_dir="output",
        lights=lights,
        background_light=env_light,
        camera=camera,
        camera_poses=camera_poses,
        samples=16,
)

# Render the dataset
render_dataset(cfg)

# Build the point cloud
pc = create_pointcloud_from_multiview("output", num_points=8192)
print("Point cloud shape:", pc.shape)  # (8192, 6)

# Visualise with Plotly
fig = create_pointcloud_figure(pc)
fig.write_html("pointcloud.html")
```

---

## License
MIT – see the [LICENSE](LICENSE) file.
