from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import *

from mesh_to_point.camera import CameraModel, CameraPose
from mesh_to_point.lights import BackgroundLight, Light


class DeviceType(str, Enum):
    CPU = "NONE"
    OPTIX = "OPTIX"
    CUDA = "CUDA"
    METAL = "METAL"
    HIP = "HIP"


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
    device_type: DeviceType = DeviceType.CPU
    samples: int = 256
    depth_pass: bool = True
