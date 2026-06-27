import argparse
from pathlib import Path
import sys
import numpy as np

from mesh_to_point.camera import read_camera_config
from mesh_to_point.io.data import write_ply
from mesh_to_point.lights import read_light_config
from mesh_to_point.pointcloud.misc import create_pointcloud_figure
from mesh_to_point.render.render import render_dataset
from mesh_to_point.pointcloud.multiview import create_pointcloud_from_multiview
from mesh_to_point.render.config import DeviceType, GlobalConfig

ASSETS_DIR = Path(__file__).parent.parent / "assets"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a mesh and generate a pointcloud."
    )
    parser.add_argument(
        "--mesh", required=True, type=Path, help="Path to the mesh file."
    )
    parser.add_argument(
        "--cameras",
        required=False,
        type=Path,
        default=ASSETS_DIR / "cameras.json",
        help="Path to the camera configuration JSON (e.g. transforms.json).",
    )
    parser.add_argument(
        "--lights",
        required=False,
        type=Path,
        default=ASSETS_DIR / "lights.json",
        help="Path to the lights configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for rendered images and pointcloud output. If omitted, it will default to `output`.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=50000,
        help="Number of points in the pointcloud.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of samples for Blender Cycles rendering.",
    )
    parser.add_argument(
        "--device-type",
        type=DeviceType,
        default=DeviceType.CPU,
        choices=[c.name for c in DeviceType],
        help=f"GPU rendering device type. Defaults to CPU usage.",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Skip pointcloud generation after rendering.",
    )
    parser.add_argument(
        "--as-ply",
        action="store_true",
        help="Store the pointcloud as text PLY file, otherwise the pointcloud is stored as a numpy (`.npy`) Nx6 array.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Stora an HTML pointcloud visualization alongside the pointcloud.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir or Path("output")
    if args.output_dir is None:
        print(f"Warning: No output directory specified; using directory {output_dir}")

    if output_dir.exists():
        print(f"Error: Output directory '{output_dir}' already exist.")
        sys.exit(1)

    # Validate that required input files exist before proceeding
    if not args.mesh.exists():
        print(f"Error: Mesh file '{args.mesh}' does not exist.")
        sys.exit(1)

    if not args.cameras.exists():
        print(f"Error: Camera configuration file '{args.cameras}' does not exist.")
        sys.exit(1)

    if not args.lights.exists():
        print(f"Error: Lights configuration file '{args.lights}' does not exist.")
        sys.exit(1)

    # Load camera and lights configuration
    try:
        camera, camera_poses = read_camera_config(args.cameras)
    except Exception as e:
        print(f"Error: There was an error during the camera configuration loading: {e}")
        sys.exit(1)

    try:
        background_light, lights = read_light_config(args.lights)
    except Exception as e:
        print(f"Error: There was an error during the lights configuration loading: {e}")
        sys.exit(1)

    # Build global configuration for rendering
    cfg = GlobalConfig(
        mesh=args.mesh,
        output_dir=output_dir,
        lights=lights,
        background_light=background_light,
        camera=camera,
        camera_poses=camera_poses,
        device_type=args.device_type,
        samples=args.samples,
        depth_pass=True,
    )

    print("Rendering dataset…")
    render_dataset(cfg)

    # After rendering, generate pointcloud from the rendered images
    if not args.no_pointcloud:
        print("Creating pointcloud...")
        pointcloud = create_pointcloud_from_multiview(
            multiview_rgb_path=output_dir,
            num_points=args.points,
            random_subsample_count=2**18,
        )

        # If specified, add plotly html visualization
        if args.html:
            fig = create_pointcloud_figure(pointcloud)
            fig.write_html(output_dir / "pointcloud.html")

        # Save pointcloud to a .npy file
        if args.as_ply:
            pc_path = output_dir / "pointcloud.ply"
            write_ply(pc_path, pointcloud=pointcloud)
        else:
            pc_path = output_dir / "pointcloud.npy"
            np.save(pc_path, pointcloud)

        print(f"Pointcloud saved to {pc_path}")
    else:
        print("Skipping pointcloud generation as requested.")


if __name__ == "__main__":
    main()
