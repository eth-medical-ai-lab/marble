"""Exact Blox environment presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from data_paths import repo_data_subpath


VALID_BLOX_CONFIG_MODES = (
    "single_turn_offline",
    "multi_turn_conversation",
)


@dataclass(frozen=True)
class BloxEnvConfig:
    """Blox environment and observation settings."""

    name: str = ""
    load_dir: Optional[str] = None
    max_steps: int = 16
    move_budget: Optional[int] = None
    with_image: bool = True
    with_symbolic: bool = True
    minimal_obs: bool = False


BLOX_BASE_CONFIG = BloxEnvConfig(
    max_steps=16,
    move_budget=None,
    with_image=True,
    with_symbolic=True,
    minimal_obs=False,
)


BLOX_ENV_CONFIGS: Dict[str, BloxEnvConfig] = {
    "easy": replace(
        BLOX_BASE_CONFIG,
        name="easy",
        load_dir=repo_data_subpath("blox", "easy"),
        max_steps=16,
        with_symbolic=True,
    ),
    "hard": replace(
        BLOX_BASE_CONFIG,
        name="hard",
        load_dir=repo_data_subpath("blox", "hard"),
        max_steps=24,
        with_symbolic=False,
    ),
}

BLOX_ENV_CONFIG_CHOICES = tuple(BLOX_ENV_CONFIGS.keys())


def _validate_blox_config_mode(mode: str) -> None:
    if mode not in VALID_BLOX_CONFIG_MODES:
        raise ValueError(
            "Unknown Blox config mode "
            f"'{mode}'. Expected one of: {', '.join(VALID_BLOX_CONFIG_MODES)}"
        )


def get_blox_env_config(name: str, mode: str) -> BloxEnvConfig:
    _validate_blox_config_mode(mode)
    try:
        return BLOX_ENV_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Blox config '{name}'. Valid configs: {', '.join(BLOX_ENV_CONFIG_CHOICES)}"
        ) from exc
