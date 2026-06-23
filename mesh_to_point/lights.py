from dataclasses import dataclass
import json
from pathlib import Path
from typing import *
from enum import Enum


class LightConfigKey(str, Enum):
    """Enum of allowed top‑level keys in a light configuration file."""

    ENVIRONMENT_COLOR = "environment_color"
    ENVIRONMENT_INTENSITY = "environment_intensity"
    LIGHTS = "lights"


class LightEntryKey(str, Enum):
    """Enum of allowed keys for each light entry in the configuration."""

    ORIGIN = "origin"
    COLOR = "color"
    INTENSITY = "intensity"
    USE_SHADOWS = "use_shadows"


@dataclass
class Light:
    origin: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    use_shadows: bool = True


class BackgroundLight:
    color: tuple[float, float, float]
    intensity: float


def read_light_config(lights_file: str | Path) -> Tuple[BackgroundLight, List[Light]]:
    """Read a JSON light configuration file.

    The configuration file is expected to contain a top‑level dictionary with
    optional keys ``environment_color`` and ``environment_intensity`` that
    describe the ambient background lighting, and a ``lights`` list that
    describes individual point lights. Each entry must contain the following
    fields: ``origin``, ``color``, and ``intensity``.

    Parameters
    ----------
    lights_file:
        Path to the JSON file.  It can be a string or a :class:`pathlib.Path`
        instance.

    Returns
    -------
    Tuple[BackgroundLighting, List[Light]]
        A tuple containing a :class:`BackgroundLighting` instance and a list
        of :class:`Light` objects parsed from the file.
    """

    # Resolve the path and open the JSON configuration file
    path = Path(lights_file)
    with path.open("r", encoding="utf-8") as f:
        light_config = json.load(f)

    # Validate that the configuration only contains known top‑level keys
    allowed_top_keys = {k.value for k in LightConfigKey}
    if not set(light_config.keys()).issubset(allowed_top_keys):
        unknown = set(light_config.keys()) - allowed_top_keys
        raise ValueError(f"Unknown top‑level keys in light config: {sorted(unknown)}")

    # Extract background lighting information
    env_color = light_config.get("environment_color", [1.0, 1.0, 1.0])
    env_intensity = light_config.get("environment_intensity", 0.0)
    background = BackgroundLight()
    background.color = tuple(env_color)
    background.intensity = float(env_intensity)

    # Build the list of Light objects
    lights: List[Light] = []
    for light_dict in light_config.get("lights", []):

        # Validate that the light entry only contains known keys
        allowed_keys = {k.value for k in LightEntryKey}
        if not set(light_dict.keys()).issubset(allowed_keys):
            unknown = set(light_dict.keys()) - allowed_keys
            raise ValueError(f"Unknown keys in light entry: {sorted(unknown)}")

        # ``origin`` is required; raise an informative error if missing
        if LightEntryKey.ORIGIN.value not in light_dict:
            raise KeyError(
                f"Missing required '{LightEntryKey.ORIGIN.value}' field in light configuration"
            )

        origin = light_dict["origin"]
        color = tuple(light_dict.get("color", [1.0, 1.0, 1.0]))
        intensity = float(light_dict.get("intensity", 1.0))
        use_shadows = bool(light_dict.get("use_shadows", True))
        lights.append(
            Light(
                origin=origin, color=color, intensity=intensity, use_shadows=use_shadows
            )
        )

    return background, lights
