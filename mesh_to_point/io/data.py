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
