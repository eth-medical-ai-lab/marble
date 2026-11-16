"""
Evaluation script for the Blox planning benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_loop import AgentLoop
from blox.config import BLOX_ENV_CONFIG_CHOICES, get_blox_env_config
from blox.agent_env import BloxAgentEnv
from blox.env import GameState
from eval_utils import positive_int, nonnegative_int
from eval_utils import ModelConfig, aggregate_usage, print_run_summary, variant_from_flags
from data_paths import public_data_path, dataset_provenance, resolve_repo_data_path
from llm_backends import LLMBackend, make_backend


def _ensure_blox_data(load_dir: str) -> None:
    """Require prebuilt Blox level data in the expected layout."""
    if os.path.isfile(os.path.join(load_dir, "samples.jsonl")):
        return
    if os.path.isdir(load_dir):
        has_json = any(
            fn.endswith(".json") and fn != "manifest.json"
            for _, _, files in os.walk(load_dir)
            for fn in files
        )
        if has_json:
            return
    raise FileNotFoundError(
        f"Expected Blox planning data under {public_data_path(load_dir)}. "
        "Use --data_root PATH or set MARBLE_DATA_ROOT to the local dataset."
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _load_blox_items(load_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load released Blox JSONL data, with legacy JSON-folder fallback."""
    jsonl_path = os.path.join(load_dir, "samples.jsonl")
    if os.path.isfile(jsonl_path):
        items = []
        base_id = public_data_path(load_dir)
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                level_id = f"{base_id}/{record.get('id', f'level_{idx:04d}')}"
                items.append((level_id, record))
        return items

    items = []
    for root, _, files in os.walk(load_dir):
        for fn in files:
            if fn.endswith(".json") and fn != "manifest.json":
                level_path = os.path.join(root, fn)
                with open(level_path, "r", encoding="utf-8") as handle:
                    items.append((public_data_path(level_path), json.load(handle)))
    items.sort(key=lambda item: item[0])
    return items


def _default_blox_max_steps(args: argparse.Namespace) -> int:
    """Default planning budget for the selected Blox config."""
    return int(args.env_config.max_steps)


def evaluate_level(
    args: argparse.Namespace,
    level_id: str,
    level_data: Dict[str, Any],
    backend: Optional[LLMBackend] = None,
) -> Tuple[bool, Dict]:
    """Evaluate a single Blox level."""
    state = GameState.from_dict(level_data)
    target_depth = level_data.get("target_depth")
    gt_solution = level_data.get("solution")
    seed = level_data.get("seed")

    dataset_name = args.dataset_name
    model_dir = str(args.model).replace("/", "-")
    variant = variant_from_flags(not args.no_image, not args.no_symbolic_obs)
    mode_dir = "offline" if args.planning_mode == "single_shot" else "online"
    base_dir = os.path.join(args.output_dir, "blox", "planning", dataset_name, model_dir, mode_dir, variant)
    os.makedirs(base_dir, exist_ok=True)

    level_name = _safe_name(level_id.split("/")[-1])
    depth_dir = _safe_name(args.dataset_name)
    run_dir = os.path.join(base_dir, "puzzles", depth_dir, level_name)
    os.makedirs(run_dir, exist_ok=True)

    cached_result = os.path.join(run_dir, "result.json")
    if os.path.exists(cached_result):
        print(f"  Cached result found, skipping: {cached_result}")
        with open(cached_result, "r", encoding="utf-8") as f:
            cached = json.load(f)
        for key in ("level_" + "path", "p" + "ath"):
            cached.pop(key, None)
        cached["level_id"] = level_id
        return bool(cached.get("success", False)), cached

    if args.planning_mode == "single_shot":
        base_env_cfg = args.env_config
        max_steps = _default_blox_max_steps(args)
        move_budget = None
        if args.move_tolerance is not None and target_depth is not None:
            move_budget = target_depth + args.move_tolerance
    else:
        base_env_cfg = args.env_config
        move_budget = None
        if args.turns is not None:
            max_steps = args.turns
        elif args.move_tolerance is not None and target_depth is not None:
            max_steps = target_depth + args.move_tolerance
        else:
            max_steps = _default_blox_max_steps(args)

    model_cfg = ModelConfig(
        model=args.model,
        local_server_port=getattr(args, "local_server_port", None),
        local_server_url=getattr(args, "local_server_url", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        max_tokens=getattr(args, "max_tokens", 32768),
    )
    env_cfg = replace(
        base_env_cfg,
        max_steps=max_steps,
        move_budget=move_budget,
    )
    agent_env = BloxAgentEnv(
        env_cfg,
        level_data=level_data,
        level_id=level_id,
        loop_mode="single_turn_offline" if args.planning_mode == "single_shot" else "multi_turn_conversation",
    )
    loop = AgentLoop(model_cfg, backend=backend)
    result = loop.run(agent_env, run_dir=run_dir)

    enriched = dict(result)
    enriched.update(
        {
            "planning_mode": args.planning_mode,
            "target_depth": target_depth,
            "gt_solution": gt_solution,
            "seed": seed,
            "board_rows": int(state.board.rows),
            "board_cols": int(state.board.cols),
        }
    )
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    usage = enriched.get("usage")
    if isinstance(usage, list):
        usage = aggregate_usage([step.get("usage") for step in usage if isinstance(step, dict)])

    if args.planning_mode == "single_shot":
        predicted_actions = list(enriched.get("predicted_actions", []))
        num_predicted = int(enriched.get("num_predicted", len(predicted_actions)))
    else:
        predicted_actions = list(enriched.get("actions_taken", []))
        num_predicted = int(enriched.get("num_steps", len(predicted_actions)))

    meta = {
        "target_depth": target_depth,
        "seed": seed,
        "num_predicted": num_predicted,
        "predicted_actions": predicted_actions,
        "usage": usage,
    }
    return bool(enriched.get("success", False)), meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LLMs on Blox benchmark")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument(
        "--config",
        type=str,
        choices=BLOX_ENV_CONFIG_CHOICES,
        required=True,
        help="Named Blox config from blox/config.py.",
    )
    ap.add_argument("--planning_mode", type=str, default="multi_step", choices=["multi_step", "single_shot"])
    ap.add_argument("--max_samples", type=nonnegative_int, default=None, help="Max levels to evaluate")
    ap.add_argument("--start_index", type=nonnegative_int, default=0, help="Zero-based level index to start from")
    ap.add_argument("--max_steps", "--turns", dest="turns", type=positive_int, default=None, help="Max steps for multi_step mode")
    ap.add_argument("--num_workers", type=positive_int, default=1, help="Number of parallel workers")
    ap.add_argument("--local_server_port", type=int, default=None)
    ap.add_argument(
        "--local_server_url",
        dest="local_server_url",
        type=str,
        default=None,
        help="Base URL of a running OpenAI-compatible inference server "
             "(for example http://host:8000/v1). Overrides --local_server_port if both are set.",
    )
    ap.add_argument("--reasoning_effort", type=str, default=None)
    ap.add_argument("--max_tokens", type=positive_int, default=32768, help="Max completion tokens per response")
    ap.add_argument(
        "--move_tolerance",
        type=int,
        default=None,
        help="Extra budget above source depth: caps a single-shot plan, or sets multi-step turns unless --max_steps is given.",
    )
    ap.add_argument("--output_dir", type=str, default="eval/output")
    args = ap.parse_args()
    args.env_config = get_blox_env_config(
        args.config,
        "single_turn_offline" if args.planning_mode == "single_shot" else "multi_turn_conversation",
    )
    preset_env_cfg = args.env_config
    args.dataset_name = args.env_config.name
    args.load_dir = str(resolve_repo_data_path(args.env_config.load_dir))
    args.no_image = not preset_env_cfg.with_image
    args.no_symbolic_obs = not preset_env_cfg.with_symbolic
    args.minimal_obs = preset_env_cfg.minimal_obs

    _ensure_blox_data(args.load_dir)

    level_items = _load_blox_items(args.load_dir)

    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.start_index:
        level_items = level_items[args.start_index:]

    if args.max_samples is not None:
        level_items = level_items[:args.max_samples]

    print(f"Found {len(level_items)} levels in {public_data_path(args.load_dir)}")

    total = 0
    successes = 0
    results = []

    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {
                ex.submit(evaluate_level, args, level_id, level_data): level_id
                for level_id, level_data in level_items
            }
            for fut in tqdm(as_completed(futs), total=len(futs)):
                level_id = futs[fut]
                try:
                    success, meta = fut.result()
                except Exception as exc:
                    print(f"Error on {level_id}: {exc}")
                    success, meta = False, {"target_depth": None, "seed": None, "num_predicted": 0, "predicted_actions": [], "usage": None}
                total += 1
                successes += int(success)
                results.append({"level_id": level_id, "success": bool(success), **meta})
    else:
        backend = make_backend(
            args.model,
            local_server_port=args.local_server_port,
            local_server_url=args.local_server_url,
        )
        for level_id, level_data in tqdm(level_items):
            try:
                success, meta = evaluate_level(args, level_id, level_data, backend=backend)
            except Exception as exc:
                print(f"Error on {level_id}: {exc}")
                success, meta = False, {"target_depth": None, "seed": None, "num_predicted": 0, "predicted_actions": [], "usage": None}
            total += 1
            successes += int(success)
            results.append({"level_id": level_id, "success": bool(success), **meta})

    dataset_name = args.dataset_name
    model_dir = str(args.model).replace("/", "-")
    variant = variant_from_flags(not args.no_image, not args.no_symbolic_obs)
    mode_dir = "offline" if args.planning_mode == "single_shot" else "online"
    base_dir = os.path.join(args.output_dir, "blox", "planning", dataset_name, model_dir, mode_dir, variant)
    os.makedirs(base_dir, exist_ok=True)

    summary = {
        "dataset": dataset_provenance(args.load_dir),
        "total": total,
        "successes": successes,
        "accuracy": successes / total if total > 0 else 0.0,
        "model": args.model,
        "config": args.config,
        "planning_mode": args.planning_mode,
        "variant": variant,
        "total_usage": aggregate_usage([r.get("usage") for r in results]),
        "results": results,
    }

    with open(os.path.join(base_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_run_summary(
        task="blox",
        difficulty=dataset_name,
        model=args.model,
        mode=mode_dir,
        variant=variant,
        successes=successes,
        total=total,
        usage=summary["total_usage"],
        output_path=os.path.join(base_dir, "results.json"),
    )


if __name__ == "__main__":
    main()
