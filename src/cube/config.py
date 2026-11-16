"""Exact Cube environment presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from data_paths import repo_data_subpath


VALID_CUBE_CONFIG_MODES = (
    "single_turn_offline",
    "multi_turn_conversation",
)


@dataclass(frozen=True)
class CubeEnvConfig:
    """Cube environment and observation settings."""

    name: str = ""
    load_dir: Optional[str] = None
    max_steps: int = 4
    with_image: bool = True
    feedback_flags: Tuple[str, ...] = ("views",) #("net", "view", "views")
    with_conflicts: bool = True
    missing: int = -1
    symbolic: bool = True
    seed: int = 20


CUBE_BASE_CONFIG = CubeEnvConfig(
    with_image=True,
    feedback_flags=("views",),
    with_conflicts=True,
    symbolic=True,
    seed=20,
)


CUBE_ENV_CONFIGS: Dict[str, CubeEnvConfig] = {
    "easy": replace(
        CUBE_BASE_CONFIG,
        name="easy",
        load_dir=repo_data_subpath("cube", "easy"),
        max_steps=4,
        missing=2,
        symbolic=True,
    ),
    "hard": replace(
        CUBE_BASE_CONFIG,
        name="hard",
        load_dir=repo_data_subpath("cube", "hard"),
        max_steps=8,
        missing=4,
        symbolic=False,
        seed=1020,
    ),
}

CUBE_ENV_CONFIG_CHOICES = tuple(CUBE_ENV_CONFIGS.keys())


def _validate_cube_config_mode(mode: str) -> None:
    if mode not in VALID_CUBE_CONFIG_MODES:
        raise ValueError(
            "Unknown Cube config mode "
            f"'{mode}'. Expected one of: {', '.join(VALID_CUBE_CONFIG_MODES)}"
        )


def get_cube_env_config(name: str, mode: str) -> CubeEnvConfig:
    _validate_cube_config_mode(mode)
    try:
        return CUBE_ENV_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Cube config '{name}'. Valid configs: {', '.join(CUBE_ENV_CONFIG_CHOICES)}"
        ) from exc
