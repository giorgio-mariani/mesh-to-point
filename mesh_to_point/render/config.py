from dataclasses import dataclass
from pathlib import Path
from typing import *

from mesh_to_point.camera import CameraModel, CameraPose


@dataclass
class Light:
    origin: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    use_shadows: bool = True


@dataclass
class GlobalConfig:
    mesh: Path
    output_dir: Path
    lights: List[Light]
    camera: CameraModel
    camera_poses: List[CameraPose]
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    background_strength: float = 1.0
    force_alpha: bool = False
    force_alpha_value: float = 1.0
    use_gpu: bool = True
    samples: int = 256
    depth_pass: bool = False
