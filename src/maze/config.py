"""Exact Maze environment presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from data_paths import repo_data_subpath


VALID_MAZE_CONFIG_MODES = (
    "single_turn_offline",
    "multi_turn_conversation",
)


@dataclass(frozen=True)
class MazeEnvConfig:
    """Maze environment and observation settings."""

    name: str = ""
    load_dir: Optional[str] = None
    max_steps: int = 8
    with_image: bool = True
    include_symbolic_obs: bool = True


MAZE_ENV_CONFIGS: Dict[str, MazeEnvConfig] = {
    "easy": MazeEnvConfig(
        name="easy",
        load_dir=repo_data_subpath("maze", "easy"),
        max_steps=6,
        include_symbolic_obs=True,
    ),
    "hard": MazeEnvConfig(
        name="hard",
        load_dir=repo_data_subpath("maze", "hard"),
        max_steps=12,
        include_symbolic_obs=False,
    ),
}

MAZE_ENV_CONFIG_CHOICES = tuple(MAZE_ENV_CONFIGS.keys())


def _validate_maze_config_mode(mode: str) -> None:
    if mode not in VALID_MAZE_CONFIG_MODES:
        raise ValueError(
            "Unknown Maze config mode "
            f"'{mode}'. Expected one of: {', '.join(VALID_MAZE_CONFIG_MODES)}"
        )


def get_maze_env_config(name: str, mode: str) -> MazeEnvConfig:
    _validate_maze_config_mode(mode)
    try:
        return MAZE_ENV_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Maze config '{name}'. Valid configs: {', '.join(MAZE_ENV_CONFIG_CHOICES)}"
        ) from exc
