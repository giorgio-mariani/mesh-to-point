from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class CameraConfig:
    origin: tuple[float, float, float]
    fov: float
    x: tuple[float, float, float]
    y: tuple[float, float, float]
    z: tuple[float, float, float]

@dataclass
class LightConfig:
    origin: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    use_shadows: bool = True

@dataclass
class ViewConfig:
    camera: CameraConfig
    resolution: tuple[int, int]
    depth: Optional[Path] = None
    rgba: Optional[Path] = None
    mask: Optional[Path] = None
    normals: Optional[Path] = None

@dataclass
class GlobalConfig:
    mesh: Path
    output_dir: Path
    lights: List[LightConfig]
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    background_strength: float = 1.0
    force_alpha: bool = False
    force_alpha_value: float = 1.0
    use_gpu: bool = True
    samples: int = 256