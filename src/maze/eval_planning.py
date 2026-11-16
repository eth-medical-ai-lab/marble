from __future__ import annotations
import os
import sys
import argparse
import json
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from agent_loop import AgentLoop
from maze.env import make_env
from maze.env import GameState
from maze.config import MAZE_ENV_CONFIG_CHOICES, get_maze_env_config
from maze.agent_env import MazeAgentEnv
from eval_utils import positive_int, nonnegative_int
from eval_utils import ModelConfig
from llm_backends import make_backend, LLMBackend
from eval_utils import variant_from_flags, aggregate_usage, print_run_summary
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from data_paths import public_data_path, dataset_provenance, resolve_repo_data_path


def _ensure_maze_data(load_dir: str) -> None:
    """Require prebuilt Maze planning data in the expected layout."""
    if os.path.isfile(os.path.join(load_dir, "samples.jsonl")):
        return
    if os.path.isdir(load_dir):
        has_json = any(
            fn.endswith('solution_trajectory.json')
            for _, _, files in os.walk(load_dir)
            for fn in files
        )
        if has_json:
            return
    raise FileNotFoundError(
        f"Expected Maze planning data under {public_data_path(load_dir)}. "
        "Use --data_root PATH or set MARBLE_DATA_ROOT to the local dataset."
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _load_maze_items(load_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load released Maze JSONL data, with legacy JSON-folder fallback."""
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
                puzzle_id = f"{base_id}/{record.get('id', f'puzzle_{idx:04d}')}"
                items.append((puzzle_id, record))
        return items

    items = []
    for root, _, files in os.walk(load_dir):
        for fn in files:
            if fn.endswith('solution_trajectory.json'):
                json_path = os.path.join(root, fn)
                with open(json_path, "r", encoding="utf-8") as handle:
                    items.append((public_data_path(json_path), json.load(handle)))
    items.sort(key=lambda item: item[0])
    return items


def _variant_for_run(args) -> str:
    return variant_from_flags(
        not args.no_image,
        not args.no_symbolic_obs,
        has_harness=bool(args.harness),
    )


def _default_multi_step_turns(args: argparse.Namespace) -> int:
    return int(args.env_config.max_steps)


def _build_env_and_obs(args, record: Dict[str, Any]) -> Tuple:
    env = make_env(start_pos=(args.start_x, args.start_y), render_mode=None, num_pieces=args.num_pieces, show_harness=args.harness, show_guides=args.guides, texture_opacity=args.texture_opacity)
    gs = GameState.from_dict(record)
    _data = record
    target = _data.get('target_piece', None)
    env.set_state_from_gs(gs, target_label=target)
    
    if args.num_pieces is not None:
        try:
            objs = dict(env._board.objects)
            tgt = getattr(env, '_target_label', None)
            others = [k for k in sorted(objs.keys()) if k != tgt]
            keep = set()
            if tgt in objs:
                keep.add(tgt)
            if args.num_pieces > 0:
                keep.update(others[: args.num_pieces])
            env._board.objects = {k: objs[k] for k in keep}
        except Exception:
            pass
    
    info_local = {"insertion_mask": env._action_mask(), "movement_mask": env._movement_mask()}
    meta = {
        "depth": _data.get('depth'),
        "initial_manhattan": _data.get('initial_manhattan'),
        "connected_path_len": _data.get('connected_path_len'),
        "seed": _data.get('seed'),
        "target_piece": target,
    }
    
    return env, info_local, meta


def _make_run_dir(args, puzzle_id: str, env, meta: dict):
    dataset_name = args.dataset_name
    mode_dir = 'offline' if args.planning_mode == 'single_shot' else 'online'
    model_dir = str(args.model).replace('/', '-')
    variant = _variant_for_run(args)
    base_dir = os.path.join(args.output_dir, "maze", "planning", dataset_name, model_dir, mode_dir, variant)

    if args.texture_opacity != 0.2:
        base_dir = os.path.join(base_dir, f"texture_opacity_{args.texture_opacity}")

    puzzle_dir_name = _safe_name(puzzle_id.split("/")[-1])
    seed_for_name = meta.get('seed') if meta.get('seed') is not None else 0
    target_label = getattr(env, '_target_label', None) or meta.get('target_piece') or 'NA'
    run_dir_local = os.path.join(base_dir, "puzzles", puzzle_dir_name, f"{seed_for_name}_{target_label}")
    os.makedirs(run_dir_local, exist_ok=True)
    
    return run_dir_local


def evaluate_path(args, puzzle_id: str, record: Dict[str, Any], backend: Optional[LLMBackend] = None):
    print(f"Evaluating {puzzle_id} with backend: {backend}")
    env, info, meta = _build_env_and_obs(args, record)
    meta_depth = meta.get('depth')
    meta_manhattan = meta.get('initial_manhattan')
    meta_connected = meta.get('connected_path_len')
    meta_seed = meta.get('seed')
    print(f"Loaded GameState from {puzzle_id} | player={env._gs.players.get('P1')} | target={getattr(env, '_target_label', None)}")
    
    model_cfg = ModelConfig(
        model=args.model,
        verbose=getattr(args, 'verbose', False),
        local_server_port=getattr(args, 'local_server_port', None),
        local_server_url=getattr(args, 'local_server_url', None),
        reasoning_effort=getattr(args, 'reasoning_effort', None),
        max_tokens=getattr(args, 'max_tokens', 32768),
    )
    run_dir_local = _make_run_dir(args, puzzle_id, env, meta)

    # Check for cached result
    cached_result = os.path.join(run_dir_local, "result.json")
    if os.path.exists(cached_result):
        print(f"  Cached result found, skipping: {cached_result}")
        with open(cached_result) as f:
            cached = json.load(f)
        return cached.get("success", False), cached

    if args.planning_mode == 'single_shot':
        base_env_cfg = args.env_config
        max_steps = int(args.turns) if args.turns is not None else int(base_env_cfg.max_steps)
    else:
        base_env_cfg = args.env_config
        if args.turns is None:
            max_steps = _default_multi_step_turns(args)
            print(f"Depth {meta_depth}, setting turns to {max_steps}")
        else:
            max_steps = int(args.turns)

    env_cfg = replace(
        base_env_cfg,
        max_steps=max_steps,
    )
    agent_env = MazeAgentEnv(
        env_cfg,
        env=env,
        max_steps=max_steps,
        info=info,
        loop_mode="single_turn_offline" if args.planning_mode == "single_shot" else "multi_turn_conversation",
    )
    loop = AgentLoop(model_cfg, backend=backend)
    result = loop.run(agent_env, run_dir=run_dir_local)
    success = bool(result.get("success", False))
    plan_list = result.get("plan")
    num_turns = result.get("num_turns")
    if num_turns is None:
        num_turns = result.get("num_steps")
    usage = result.get("usage")
    if isinstance(usage, list):
        usage = aggregate_usage([item.get("usage") for item in usage if isinstance(item, dict)])

    print("\n" + ("✅ SUCCESS: Target reached." if success else "❌ FAILED: Target not reached."))

    meta = {
        "depth": meta_depth,
        "initial_manhattan": meta_manhattan,
        "connected_path_len": meta_connected,
        "seed": meta_seed,
        "plan": plan_list,
        "num_turns": num_turns,
        "usage": usage,
    }

    # Save per-puzzle result.json (consistent with cube/blox)
    if run_dir_local:
        result = dict(result)
        result.update({"puzzle_id": puzzle_id, **meta})
        with open(os.path.join(run_dir_local, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return bool(success), meta

def main():
    ap = argparse.ArgumentParser(description="Run MLLM agent in Labyrinth env")
    ap.add_argument('--start_x', type=int, default=0)
    ap.add_argument('--start_y', type=int, default=0)
    ap.add_argument('--max_steps', '--turns', dest='turns', type=positive_int, default=None)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--config', type=str, choices=MAZE_ENV_CONFIG_CHOICES, required=True,
                    help='Named Maze config from maze/config.py')
    ap.add_argument('--planning_mode', type=str, default='multi_step', choices=['multi_step','single_shot'], help='planning prompt style')
    ap.add_argument('--num_pieces', type=int, default=0, help='number of treasures to keep besides the target (pruned deterministically)')
    ap.add_argument('--harness', action='store_true', help='overlay harness labels on the board')
    ap.add_argument('--guides', action='store_true', help='overlay guides on the board')
    ap.add_argument('--num_workers', type=positive_int, default=1, help='Number of parallel workers (1 = sequential)')
    ap.add_argument('--texture_opacity', type=float, default=0.2, help='opacity of the texture overlay')
    ap.add_argument('--local_server_port', type=int, default=None, help='Port of a running local inference server (vLLM or SGLang) to use (e.g. 8000)')
    ap.add_argument('--local_server_url', dest='local_server_url', type=str, default=None,
                    help='Base URL of a running OpenAI-compatible inference server '
                         '(for example http://host:8000/v1). Overrides --local_server_port if both are set.')
    ap.add_argument('--max_samples', type=nonnegative_int, default=None, help='limit number of puzzles to evaluate')
    ap.add_argument('--start_index', type=nonnegative_int, default=0, help='zero-based puzzle index to start from')
    ap.add_argument('--reasoning_effort', type=str, default=None)
    ap.add_argument('--max_tokens', type=positive_int, default=32768, help='Max completion tokens per model call')
    ap.add_argument('--verbose', action='store_true', default=False,
                    help='Print the model\'s raw response text to stdout')
    ap.add_argument('--output_dir', type=str, default='eval/output',
                    help='Base output directory (default: eval/output)')
    ap.set_defaults(guides=True, harness=False)
    args = ap.parse_args()
    args.env_config = get_maze_env_config(
        args.config,
        "single_turn_offline" if args.planning_mode == "single_shot" else "multi_turn_conversation",
    )
    preset_env_cfg = args.env_config
    args.dataset_name = args.env_config.name
    args.load_dir = str(resolve_repo_data_path(args.env_config.load_dir))
    args.no_image = not preset_env_cfg.with_image
    args.no_symbolic_obs = not preset_env_cfg.include_symbolic_obs

    _ensure_maze_data(args.load_dir)

    if args.load_dir:
        puzzle_items = _load_maze_items(args.load_dir)
        if args.start_index < 0:
            raise ValueError("--start_index must be non-negative")
        if args.start_index:
            puzzle_items = puzzle_items[args.start_index:]
        if args.max_samples is not None:
            puzzle_items = puzzle_items[:args.max_samples]
        print(f"Found {len(puzzle_items)} puzzles in {public_data_path(args.load_dir)}")
        total = 0
        successes = 0
        results = []
        
        if args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futs = {
                    ex.submit(evaluate_path, args, puzzle_id, record): puzzle_id
                    for puzzle_id, record in puzzle_items
                }
                
                for fut in tqdm(as_completed(futs), total=len(futs)):
                    puzzle_id = futs[fut]
                    try:
                        success, meta = fut.result()
                    except Exception as e:
                        traceback.print_exc()
                        print(f"Skipping {puzzle_id}: {e}")
                        total += 1
                        results.append({
                            "puzzle_id": puzzle_id,
                            "success": False,
                            "depth": None,
                            "initial_manhattan": None,
                            "connected_path_len": None,
                            "seed": None,
                            "num_turns": None,
                            "plan": None,
                            "usage": None,
                        })
                        continue

                    total += 1
                    successes += int(success)
                    results.append({
                        "puzzle_id": puzzle_id,
                        "success": bool(success),
                        "depth": meta.get("depth"),
                        "initial_manhattan": meta.get("initial_manhattan"),
                        "connected_path_len": meta.get("connected_path_len"),
                        "seed": meta.get("seed"),
                        "num_turns": meta.get("num_turns"),
                        "plan": meta.get("plan"),
                        "usage": meta.get("usage"),
                    })
        else:
            backend = make_backend(
                args.model,
                local_server_port=args.local_server_port,
                local_server_url=args.local_server_url,
            )
            print(f"Using backend: {backend}")
            
            for puzzle_id, record in tqdm(puzzle_items):
                try:
                    success, meta = evaluate_path(args, puzzle_id, record, backend=backend)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error evaluating {puzzle_id}: {e}")
                    success, meta = False, {"depth": None, "initial_manhattan": None, "connected_path_len": None, "seed": None, "num_turns": None, "plan": None, "usage": None}

                total += 1
                successes += int(success)
                results.append({
                    "puzzle_id": puzzle_id,
                    "success": bool(success),
                    "depth": meta.get("depth"),
                    "initial_manhattan": meta.get("initial_manhattan"),
                    "connected_path_len": meta.get("connected_path_len"),
                    "seed": meta.get("seed"),
                    "num_turns": meta.get("num_turns"),
                    "plan": meta.get("plan"),
                    "usage": meta.get("usage"),
                })
        
        dataset_name = args.dataset_name
        model_dir = str(args.model).replace('/', '-')
        mode_dir = 'offline' if args.planning_mode == 'single_shot' else 'online'
        variant = variant_from_flags(
            not args.no_image,
            not args.no_symbolic_obs,
            has_harness=bool(args.harness),
        )
        
        base_dir = os.path.join(args.output_dir, "maze", "planning", dataset_name, model_dir, mode_dir, variant)
        if args.texture_opacity != 0.2:
            base_dir = os.path.join(base_dir, f"texture_opacity_{args.texture_opacity}")
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
        with open(os.path.join(base_dir, "results.json"), 'w', encoding='utf-8') as rf:
            json.dump(summary, rf, indent=2, ensure_ascii=False)
        print_run_summary(
            task="maze", difficulty=dataset_name, model=args.model,
            mode=mode_dir, variant=variant,
            successes=successes, total=total,
            usage=summary["total_usage"],
            output_path=os.path.join(base_dir, "results.json"),
        )
        
        return

if __name__ == '__main__':
    main()
