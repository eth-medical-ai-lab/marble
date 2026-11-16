"""Small helpers for the released MARBLE data layout."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Union

from huggingface_hub import snapshot_download


PathLike = Union[str, Path]
TASK_NAMES = {"cube", "maze", "blox"}
REPO_ROOT = Path(__file__).resolve().parents[1]
# Pin the verified canonical dataset for reproducible evaluation.
HF_REPO_ID = "mrble/MARBLE"
HF_REVISION = "2cdb82398e8b11b535151551568b0c38fd118fad"
_HF_SOURCES: dict[Path, dict] = {}


def _local_data_root() -> Path:
    return Path(os.environ.get("MARBLE_DATA_ROOT") or REPO_ROOT / "data").expanduser().resolve()


def repo_data_path(task: str, *parts: str) -> Path:
    """Return an absolute path under the configured data root."""
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task '{task}'")
    return _local_data_root().joinpath(task, *parts)


def repo_data_subpath(task: str, *parts: str) -> str:
    """Return the public data path used in CLI defaults."""
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task '{task}'")
    return Path("data", task, *parts).as_posix()


def resolve_repo_data_path(path: PathLike) -> Path:
    """Load a selected canonical set from HF, or use an explicit local path/root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    parts = candidate.parts
    if len(parts) < 3 or parts[0] != "data" or parts[1] not in TASK_NAMES:
        return candidate
    if os.environ.get("MARBLE_DATA_ROOT"):
        return _local_data_root().joinpath(*parts[1:])
    if HF_REPO_ID is None or HF_REVISION is None:
        raise RuntimeError(
            "The canonical Hugging Face dataset is not published yet. "
            "Use --data_root PATH or set MARBLE_DATA_ROOT to your local dataset."
        )

    # Resolve only the selected set, including its frozen perception images.
    # One worker avoids the HF cache's first-use symlink race on Windows.
    task, split = parts[1:3]
    snapshot = Path(snapshot_download(
        repo_id=HF_REPO_ID, repo_type="dataset", revision=HF_REVISION,
        allow_patterns=f"{task}/{split}/*", max_workers=1,
    ))
    folder = (snapshot / task / split).resolve()
    _HF_SOURCES[folder] = {"repo_id": HF_REPO_ID, "revision": HF_REVISION,
                           "task": task, "split": split}
    return folder.joinpath(*parts[3:])


def public_data_path(path: PathLike) -> str:
    """Return a non-identifying path for logs and result files."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    resolved = candidate.resolve()
    for folder, source in _HF_SOURCES.items():
        if resolved.is_relative_to(folder):
            return Path("data", source["task"], source["split"]).joinpath(
                resolved.relative_to(folder)).as_posix()
    try:
        return Path("data").joinpath(resolved.relative_to(_local_data_root())).as_posix()
    except ValueError:
        return resolved.name


def load_perception_records(load_dir: PathLike, max_samples: int | None = None) -> list[dict]:
    """Read fixed held-out records and their images; never render during evaluation."""
    from PIL import Image

    path = resolve_repo_data_path(load_dir)
    sample_file = path if path.is_file() else path / "samples.jsonl"
    with sample_file.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be nonnegative")
        records = records[:max_samples]
    for record in records:
        with Image.open(sample_file.parent / record["image_path"]) as image:
            record["image"] = image.convert("RGB")
    return records


def dataset_provenance(load_dir: PathLike) -> dict:
    """Record the actual JSONL hash, manifest, and selected HF release."""
    path = resolve_repo_data_path(load_dir)
    sample_file = path if path.is_file() else path / "samples.jsonl"
    info = {}
    if sample_file.is_file():
        info["samples_sha256"] = hashlib.sha256(sample_file.read_bytes()).hexdigest()
    for name in ("manifest.json", "download.json"):
        file = sample_file.parent / name
        if file.is_file():
            info[name.removesuffix(".json")] = json.loads(file.read_text(encoding="utf-8"))
    if sample_file.parent.resolve() in _HF_SOURCES:
        info["download"] = _HF_SOURCES[sample_file.parent.resolve()].copy()
    return info
