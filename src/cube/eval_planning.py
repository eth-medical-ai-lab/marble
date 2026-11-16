import os
import sys
import random
import numpy as np
import argparse
import json
from dataclasses import replace
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cube.agent_env import extract_happycube_solution
from cube.env import normalize_cube_solution, select_locked_placements
from agent_loop import AgentLoop
from llm_backends import make_backend
from data_paths import public_data_path, dataset_provenance, resolve_repo_data_path
from eval_utils import positive_int, nonnegative_int
from eval_utils import variant_from_flags, create_text_box, aggregate_usage, print_run_summary, ModelConfig

from cube.config import CUBE_ENV_CONFIG_CHOICES, CubeEnvConfig, get_cube_env_config
from cube.agent_env import CubeAgentEnv
from cube.prompts import build_cube_prompt, format_cube_solution

load_dotenv()


def _cube_variant_from_flags(args) -> str:
    """Build variant string from observation flags, matching maze/blox convention."""
    has_image = (args.image == 'on') or (args.image == 'auto')
    has_symbolic = bool(args.symbolic)
    extra = []
    return variant_from_flags(has_image, has_symbolic, extra_suffixes=extra)


def _cube_base_dir(args) -> str:
    """Build output base dir: eval/output/cube/planning/{dataset}/{model}/{offline|online}/{variant}."""
    model_dir = args.model.replace("/", "-")
    mode_dir = "offline" if args.planning_mode == "single_shot" else "online"
    variant = _cube_variant_from_flags(args)
    return os.path.join(
        args.output_dir,
        "cube",
        "planning",
        _cube_dataset_name(args),
        model_dir,
        mode_dir,
        variant,
    )


def _cube_dataset_name(args) -> str:
    if getattr(args, "dataset_name", None):
        return str(args.dataset_name)
    if getattr(args, "load_dir", None):
        return Path(args.load_dir).resolve().name
    if getattr(args, "config", None):
        return str(args.config)
    raise ValueError("Cube run needs dataset_name, load_dir, or config to derive an output path.")


def _load_local_cube_dataset(load_dir: str) -> List[Dict[str, Any]]:
    root = Path(load_dir).resolve()
    planning_path = root / "samples.jsonl"
    if not planning_path.is_file():
        raise FileNotFoundError(f"Missing planning file: {public_data_path(planning_path)}")

    records: List[Dict[str, Any]] = []
    with planning_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "faces" not in sample and "face_arrays" in sample:
                sample["faces"] = sample["face_arrays"]
            if "face_arrays" not in sample and "faces" in sample:
                sample["face_arrays"] = sample["faces"]
            sample.setdefault("id", f"sample_{idx:04d}")
            records.append(sample)

    return records


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--config', type=str, choices=CUBE_ENV_CONFIG_CHOICES, required=True,
                    help='Named Cube config from cube/config.py.')
parser.add_argument('--num_workers', type=positive_int, default=1, help='Number of parallel workers (1 = sequential)')
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--max_samples', type=nonnegative_int, default=None, help='limit number of samples; default uses full dataset')
parser.add_argument('--start_index', type=nonnegative_int, default=0, help='zero-based sample index to start from')
parser.add_argument('--planning_mode', type=str, default='multi_step',
                    choices=['single_shot', 'multi_step'],
                    help='single_shot: single-turn offline; multi_step: agentic step-by-step')
parser.add_argument('--max_steps', type=positive_int, default=None,
                    help='Max placement steps for multi_step mode (default: 2 * missing)')
parser.add_argument('--max_tokens', type=positive_int, default=32768,
                    help='Max completion tokens per model call (default: 32768)')
parser.add_argument('--reasoning_effort', type=str, default=None)
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Print the model\'s raw response text to stdout')
parser.add_argument('--output_dir', type=str, default='eval/output',
                    help='Base output directory (default: eval/output)')
parser.add_argument('--local_server_port', type=int, default=None,
                    help='Port of a running local inference server (vLLM or SGLang) to use (e.g. 8000)')
parser.add_argument('--local_server_url', dest='local_server_url', type=str, default=None,
                    help='Base URL of a running OpenAI-compatible inference server '
                         '(for example http://host:8000/v1). Overrides --local_server_port if both are set.')
parser.add_argument('--image_mode', type=str, choices=['on', 'off', 'auto'], default=None,
                    help='Override the preset image observation mode from the selected config.')
# Importing dataset helpers must not parse CLI arguments or reset random seeds.
if __name__ == "__main__":
    args = parser.parse_args()

    args.env_config = get_cube_env_config(
        args.config,
        "single_turn_offline" if args.planning_mode == "single_shot" else "multi_turn_conversation",
    )
    args.dataset_name = args.env_config.name
    args.load_dir = str(resolve_repo_data_path(args.env_config.load_dir))
    preset_env_cfg = args.env_config
    args.missing = preset_env_cfg.missing
    args.default_image_enabled = bool(preset_env_cfg.with_image)
    args.image = 'on' if preset_env_cfg.with_image else 'off'
    args.symbolic = preset_env_cfg.symbolic
    args.conflicts = preset_env_cfg.with_conflicts
    args.feedback = []
    if any(flag in preset_env_cfg.feedback_flags for flag in ('net', 'view', 'views')):
        args.feedback.append('image')
    if 'symbolic' in preset_env_cfg.feedback_flags:
        args.feedback.append('symbolic')
    if args.image_mode is not None:
        args.image = args.image_mode

    # Derive max_steps from missing count if not explicitly set
    if args.max_steps is None:
        args.max_steps = int(preset_env_cfg.max_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)

def _use_image() -> bool:
    if args.image == 'on':
        return True
    if args.image == 'off':
        return False
    return bool(args.default_image_enabled)


def _feedback_flags() -> List[str]:
    flags: List[str] = []
    for item in args.feedback:
        if item == 'image':
            flags.append('views')
        elif item in ('net', 'view', 'views', 'symbolic'):
            flags.append(item)
    if _use_image() and not any(flag in flags for flag in ('net', 'view', 'views')):
        flags.insert(0, 'net')
    if 'symbolic' in args.feedback and 'symbolic' not in flags:
        flags.append('symbolic')
    return flags


def _build_model_cfg() -> ModelConfig:
    return ModelConfig(
        model=args.model,
        verbose=getattr(args, 'verbose', False),
        local_server_port=getattr(args, 'local_server_port', None),
        local_server_url=getattr(args, 'local_server_url', None),
        reasoning_effort=getattr(args, 'reasoning_effort', None),
        max_tokens=int(args.max_tokens),
    )


def _build_env_cfg(loop_mode: str) -> CubeEnvConfig:
    del loop_mode
    base_cfg = args.env_config
    return replace(
        base_cfg,
        with_image=_use_image(),
        feedback_flags=tuple(_feedback_flags()),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )


def _solution_to_sample_map(solution: Dict[str, Tuple[str, int, int]]) -> Dict[str, List[Any]]:
    return {
        letter: [str(face).capitalize(), int(left_edge), int(top_edge)]
        for letter, (face, left_edge, top_edge) in solution.items()
    }


def _prepare_cube_sample(
    index: int,
    sample: Dict[str, Any],
    show_prompt: bool = False,
) -> Tuple[Dict[str, Any], str]:
    sample = dict(sample)
    if "faces" not in sample and "face_arrays" in sample:
        sample["faces"] = sample["face_arrays"]

    puzzle_seed = args.seed + index
    solution = normalize_cube_solution(sample.get("solution"))
    if not solution:
        ref_sol = extract_happycube_solution(str(sample.get("reference_solution", "") or "")) or {}
        solution = {
            letter: (str(face).lower(), int(left_edge), int(top_edge))
            for letter, (face, left_edge, top_edge) in ref_sol.items()
        }

    explicit_locked = normalize_cube_solution(sample.get("locked_placements"))
    if args.missing is not None and int(args.missing) >= 0 and solution:
        locked_placements = select_locked_placements(
            solution=solution,
            desired_missing=int(args.missing),
            seed=puzzle_seed,
        )
    elif explicit_locked:
        locked_placements = explicit_locked
    elif solution:
        meta_info = sample.get("meta_info") if isinstance(sample.get("meta_info"), dict) else {}
        if "missing" not in meta_info:
            raise ValueError("Cube sample missing meta_info['missing'] and config does not define missing.")
        default_missing = int(meta_info["missing"])
        locked_placements = select_locked_placements(
            solution=solution,
            desired_missing=default_missing,
            seed=puzzle_seed,
        )
    else:
        locked_placements = {}

    sample["solution"] = _solution_to_sample_map(solution)
    sample["locked_placements"] = _solution_to_sample_map(locked_placements)
    sample["face_arrays"] = sample["faces"]

    if show_prompt:
        prompt_text = build_cube_prompt(
            face_arrays=sample["faces"],
            locked_placements=locked_placements,
            loop_mode="single_turn_offline",
            symbolic=bool(args.symbolic),
        )
        print("   🗣️ Prompt:")
        print("\n".join(create_text_box(prompt_text)))
        print("   🗣️ Full Solution:")
        print("\n".join(create_text_box(format_cube_solution(solution) if solution else "")))

    return sample, format_cube_solution(solution) if solution else ""

def call_api(model, index, sample, backend, log_file):
    del model
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)['id'] for line in f.readlines()]
        if index in ids:
            print(f'{index} already done!')
            return

    run_dir = os.path.join(os.path.dirname(log_file), "puzzles", f"puzzle_{index:04d}")

    try:
        prepared_sample, _ = _prepare_cube_sample(
            index=index,
            sample=sample,
            show_prompt=True,
        )
        model_cfg = _build_model_cfg()
        env_cfg = _build_env_cfg("single_turn_offline")
        agent_env = CubeAgentEnv(
            env_cfg,
            prepared_sample,
            sample_idx=index,
            loop_mode="single_turn_offline",
        )
        loop = AgentLoop(model_cfg, backend=backend)
        result = loop.run(agent_env, run_dir=run_dir)
        response = result.get('response', '')
        if getattr(args, 'verbose', False):
            print(f"\n{'='*60}\n[Model Response]\n{'='*60}\n{response}\n{'='*60}\n")

        print("   🤖 Response:")
        print("\n".join(create_text_box(response or "")))

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error processing sample {index}: {e}")


def get_result(log_file):
    with open(log_file, 'r') as f:
        accuracies = [bool(json.loads(line)['correct']) for line in f.readlines()]
    total = len(accuracies)
    correct = int(sum(accuracies))
    return {
        'accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
        'correct': correct,
        'total': total,
    }


if __name__ == '__main__':
    if not args.load_dir:
        raise ValueError(
            f"Cube config '{args.config}' must set load_dir. Hugging Face fallback has been removed."
        )
    using_local_dataset = True
    dataset = _load_local_cube_dataset(args.load_dir)
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.start_index:
        dataset = dataset[args.start_index:]
    if args.max_samples is not None and args.max_samples >= 0:
        dataset = dataset[:min(args.max_samples, len(dataset))]
    print(f"Loaded {len(dataset)} local Cube samples from {public_data_path(args.load_dir)}")

    backend = make_backend(
        args.model,
        local_server_port=args.local_server_port,
        local_server_url=args.local_server_url,
    )

    if args.planning_mode == 'multi_step':
        # ----- Multi-step agentic evaluation -----
        base_dir = _cube_base_dir(args)
        os.makedirs(base_dir, exist_ok=True)

        total = 0
        successes = 0
        results_list = []

        print(f"Multi-step evaluation: {len(dataset)} puzzles, max_steps={args.max_steps}, "
              f"num_workers={args.num_workers}")

        def _run_one(i, sample, per_thread_backend=None):
            b = per_thread_backend or backend
            run_dir = os.path.join(base_dir, "puzzles", f"puzzle_{i:04d}")
            try:
                if using_local_dataset:
                    prepared_sample = dict(sample)
                else:
                    prepared_sample, _ = _prepare_cube_sample(
                        index=i,
                        sample=sample,
                        show_prompt=False,
                    )
                model_cfg = _build_model_cfg()
                env_cfg = _build_env_cfg("multi_turn_conversation")
                agent_env = CubeAgentEnv(
                    env_cfg,
                    prepared_sample,
                    sample_idx=i,
                    loop_mode="multi_turn_conversation",
                )
                loop = AgentLoop(model_cfg, backend=b)
                result = loop.run(agent_env, run_dir=run_dir)
            except Exception as e:
                print(f"Error on puzzle {i}: {e}")
                result = {"success": False, "actions_taken": [], "num_steps": 0, "usage": []}

            success = result.get("success", False)
            meta = {
                "num_steps": result.get("num_steps", 0),
                "actions_taken": result.get("actions_taken", []),
                "step_usages": [s["usage"] for s in result.get("usage", [])],
            }
            return i, success, meta

        if args.num_workers > 1:
            # Each worker uses its own chat client.
            worker_backends = [
                make_backend(
                    args.model,
                    local_server_port=args.local_server_port,
                    local_server_url=args.local_server_url,
                )
                for _ in range(args.num_workers)
            ]
            import itertools
            backend_cycle = itertools.cycle(worker_backends)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(_run_one, i, sample, next(backend_cycle))
                           for i, sample in enumerate(dataset, start=args.start_index)]
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    i, success, meta = fut.result()
                    total += 1
                    successes += int(success)
                    results_list.append({"id": i, "success": bool(success), **meta})
        else:
            for i, sample in tqdm(enumerate(dataset, start=args.start_index), total=len(dataset)):
                _, success, meta = _run_one(i, sample)
                total += 1
                successes += int(success)
                results_list.append({"id": i, "success": bool(success), **meta})

        # Save aggregate summary
        all_usages = [u for r in results_list for u in (r.get("step_usages") or [])]
        summary = {
            "dataset": dataset_provenance(args.load_dir),
            "total": total,
            "successes": successes,
            "accuracy": successes / total if total > 0 else 0.0,
            "total_usage": aggregate_usage(all_usages),
            "model": args.model,
            "dataset_name": _cube_dataset_name(args),
            "data_dir": public_data_path(args.load_dir),
            "config": args.config,
            "planning_mode": "multi_step",
            "max_steps": args.max_steps,
            "feedback": args.feedback,
            "conflicts": args.conflicts,
            "missing": args.missing,
            "results": results_list,
        }
        summary_path = os.path.join(base_dir, "results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        variant = _cube_variant_from_flags(args)
        difficulty = args.config
        mode_dir = "online"
        print_run_summary(
            task="cube", difficulty=difficulty, model=args.model,
            mode=mode_dir, variant=variant,
            successes=successes, total=total,
            usage=summary.get("total_usage"),
            output_path=summary_path,
        )

    else:
        # ----- Single-shot (offline) evaluation -----
        base_dir = _cube_base_dir(args)
        os.makedirs(base_dir, exist_ok=True)
        # Build filename suffixes for missing-piece config
        suffix_parts = []
        if args.missing is not None and int(args.missing) >= 0:
            suffix_parts.append(f"p_{int(args.missing)}")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        log_file = os.path.join(base_dir, f"results.jsonl")
        run_results = []

        if args.num_workers == 1:
            print("Dataset Size: ", len(dataset))
            for i, sample in tqdm(enumerate(dataset, start=args.start_index), total=len(dataset)):
                if using_local_dataset:
                    run_dir = os.path.join(base_dir, "puzzles", f"puzzle_{i:04d}")
                    model_cfg = _build_model_cfg()
                    env_cfg = _build_env_cfg("single_turn_offline")
                    agent_env = CubeAgentEnv(
                        env_cfg,
                        dict(sample),
                        sample_idx=i,
                        loop_mode="single_turn_offline",
                    )
                    loop = AgentLoop(model_cfg, backend=backend)
                    result = loop.run(agent_env, run_dir=run_dir)
                    run_results.append(result)
                else:
                    call_api(args.model, i, sample, backend, log_file)
        else:
            # Each worker uses its own client; the main thread writes the aggregate.
            import itertools
            worker_backends = [
                make_backend(
                    args.model,
                    local_server_port=args.local_server_port,
                    local_server_url=args.local_server_url,
                )
                for _ in range(args.num_workers)
            ]
            backend_cycle = itertools.cycle(worker_backends)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                if using_local_dataset:
                    def _run_single_local(i, sample, worker_backend):
                        run_dir = os.path.join(base_dir, "puzzles", f"puzzle_{i:04d}")
                        model_cfg = _build_model_cfg()
                        env_cfg = _build_env_cfg("single_turn_offline")
                        agent_env = CubeAgentEnv(
                            env_cfg,
                            dict(sample),
                            sample_idx=i,
                            loop_mode="single_turn_offline",
                        )
                        loop = AgentLoop(model_cfg, backend=worker_backend)
                        result = loop.run(agent_env, run_dir=run_dir)
                        return result

                    futures = [
                        executor.submit(
                            _run_single_local, i, sample, next(backend_cycle)
                        ) for i, sample in enumerate(dataset, start=args.start_index)
                    ]
                else:
                    futures = [
                        executor.submit(
                            call_api, args.model, i, sample, next(backend_cycle), log_file
                        ) for i, sample in enumerate(dataset, start=args.start_index)
                    ]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if using_local_dataset:
                        run_results.append(result)

        # Rebuild this selection's aggregate from cached/new per-puzzle results.
        # This also creates an empty log for zero samples and avoids duplicates on resume.
        if using_local_dataset:
            with open(log_file, "w", encoding="utf-8") as log:
                for result in sorted(run_results, key=lambda row: row["id"]):
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")

        result = get_result(log_file)

        # Aggregate token usage from per-sample results
        usage_list = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    usage_list.append(rec.get('usage'))
                except Exception:
                    continue
        total_usage = aggregate_usage(usage_list)

        # Save structured summary alongside the JSONL
        summary = {
            "dataset": dataset_provenance(args.load_dir),
            'dataset_name': _cube_dataset_name(args),
            'data_dir': public_data_path(args.load_dir),
            'config': args.config,
            'model': args.model,
            **result,
            'total_usage': total_usage,
            **({'missing_pieces': int(args.missing)} if (args.missing is not None and int(args.missing) >= 0) else {}),
        }
        summary_path = os.path.join(base_dir, f"results.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        variant = _cube_variant_from_flags(args)
        difficulty = args.config
        mode_dir = "offline"
        print_run_summary(
            task="cube", difficulty=difficulty, model=args.model,
            mode=mode_dir, variant=variant,
            successes=result.get('correct', 0), total=result.get('total', 0),
            usage=total_usage,
            output_path=summary_path,
        )

        # Also append to global ledger for backward compat
        os.makedirs(os.path.join(args.output_dir, 'cube', 'planning'), exist_ok=True)
        with open(os.path.join(args.output_dir, 'cube', 'planning', 'final_result.jsonl'), 'a') as f:
            f.write(json.dumps(summary) + '\n')

