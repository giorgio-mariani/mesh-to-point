from pathlib import Path
from typing import Tuple

import numpy as np


def load_depth_file(filepath: str) -> np.ndarray:
    import OpenEXR

    (exr_part,) = OpenEXR.File(str(filepath)).parts
    return exr_part.channels["V"].pixels.reshape(-1, 1)


def load_rgba_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    rgba = np.array(Image.open(filepath)).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return rgb.reshape(-1, 3), alpha.reshape(-1, 1)


def write_ply(output_path: str | Path, pointcloud: np.ndarray) -> None:
    """Save a point cloud as an ASCII PLY file.

    Parameters
    ----------
    output_path:
        Path to the output file. The extension is ignored – the function writes
        an ASCII PLY regardless of the supplied extension.
    pointcloud:
        A NumPy array of shape ``(N, 6)`` where the first three columns are
        ``(x, y, z)`` coordinates and the last three columns are ``(r, g, b)``
        values in the range ``[0, 1]``.
    """
    # Raise error if the output directory does not exist
    output_path = Path(output_path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory '{output_path.parent}' does not exist."
        )

    # Convert colors to 0-255 uint8
    colors = (pointcloud[:, 3:] * 255).astype(np.uint8)
    vertices = pointcloud[:, :3]

    with open(output_path, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(vertices)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("end_header\n")
        for v, c in zip(vertices, colors):
            fp.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
