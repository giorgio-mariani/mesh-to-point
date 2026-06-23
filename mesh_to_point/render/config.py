from dataclasses import dataclass
from pathlib import Path
from typing import *

from mesh_to_point.camera import CameraModel, CameraPose
from mesh_to_point.lights import BackgroundLight, Light


@dataclass
class GlobalConfig:
    mesh: Path
    output_dir: Path
    lights: List[Light]
    background_light: BackgroundLight
    camera: CameraModel
    camera_poses: List[CameraPose]
    force_alpha: bool = False
    force_alpha_value: float = 1.0
    use_gpu: bool = True
    samples: int = 256
    depth_pass: bool = True
