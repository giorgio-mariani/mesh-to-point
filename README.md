# Mesh‑to‑Point

## Overview
`mesh-to-point` is a lightweight Python library that converts 3D meshes into dense RGB point clouds.

It is built on top of **NumPy**, and **scikit‑learn** for data handling, and uses **Blender** (via the `bpy` API) for rendering synthetic views.

The project ships a small command‑line interface that can render a mesh from multiple camera viewpoints and generate a point cloud from the rendered RGB-D images.


## Installation
The package is published on PyPI and can be installed with `pip`:

```bash
pip install mesh-to-point
```

Note that this will also install a headless version of blender 5.0 to be used for rendering.

**Development install.** If you want to contribute or run the test suite, install the optional development dependencies:

```bash
pip install .[dev]
pre-commit install
```

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

**Output directory.** The output directory contains the rendered images, camera transforms, point cloud data, and an optional HTML visualization. Images are stored under ``<output_dir>/images`` as ``<index>_rgb.png`` and ``<index>_depth.exr``. Extrinsic and intrinsic camera parameters are saved in ``transforms.json``. The point cloud is written as ``pointcloud.npy`` (or ``pointcloud.ply`` if ``--as-ply`` is used). If ``--html`` is enabled, an interactive Plotly visualization is generated as ``pointcloud.html``.

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

**Configuration files.** Lighting and camera information are provided to the CLI through JSON files; These describe the extrinsic camera poses and the intrinsic camera parameters, together with the amount of point lights in the scene and the background light color and intensity. If these are not provided, `mesh-to-point` will default to [[assets/cameras.json]] and [[assets/lights.json]] for cameras and lights respectively.
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

### API reference
The library exposes a small set of public functions that can be used programmatically:

* ``mesh_to_point.render_dataset`` – Render a dataset of RGB-D images from a mesh.
* ``mesh_to_point.create_pointcloud_from_multiview`` – Build a point cloud from a directory of rendered RGB-D images.
* ``mesh_to_point.create_pointcloud_figure`` – Create a Plotly figure for visualising a point cloud.

#### Examples
If necessary, it is possible to directly invoke the python rendering function instead of using the [command-line interface](#command-line-interface). For example, by calling the `render_dataset` function directly:
```python
from mesh_to_point import render_dataset, GlobalConfig, read_camera_config, read_light_config

# Load camera and lights configuration
camera, camera_poses = read_camera_config("path-to-camera-cfg")
background_light, lights = read_light_config("path-to-light-cfg")

# Setup rendering config
cfg = GlobalConfig(
    mesh="path-to-mesh",
    output_dir="output",
    lights=lights,
    background_light=background_light,
    camera=camera,
    camera_poses=camera_poses,
    use_gpu=True,
    samples=16,
    depth_pass=True,
)

# Render views
render_dataset(cfg)
```
Once the multiple views have been rendered, it is possible to construct a point cloud from them by
using the `create_pointcloud_from_multiview` utility:
```python
from mesh_to_point import create_pointcloud_from_multiview

# Path to the folder that contains the rendered images.
# The folder must follow the CLI output layout:
#        <output_dir>/images/<idx>_rgb.png
#        <output_dir>/images/<idx>_depth.exr
render_dir = "output"
pointcloud = create_pointcloud_from_multiview(
    render_dir, num_points=8192,
)

print(f"Point cloud shape: {pointcloud.shape}")   # (8192, 6)

```
Finally, it is also possible to produce an HTML/javascript file to visualize the point cloud on a browser of your choice.
```python
from mesh_to_point import create_pointcloud_figure
# Visualise the point cloud with Plotly.
fig = create_pointcloud_figure(pointcloud)
fig.write_html("pointcloud.html")   # open in a browser to visualize the point cloud
```

## License
MIT – see the [LICENSE](LICENSE) file.
