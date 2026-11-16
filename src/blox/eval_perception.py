"""
Perception evaluation for Blox: board state parsing from isometric 3D renders.

Given an image of a Blox level, extract:
  - Grid layout (rows x cols array of ./#/G/F/C/S/~/=/E)
  - Block position (row, col) and orientation
  - Goal position (row, col)
  - Crumble tile positions
  - Elevator cells (positions of elevator tiles)
  - Elevated cells (positions on the HIGH level)
  - Block height (LOW/HIGH)
  - Height map (per-cell height level)
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import glob
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_paths import public_data_path, repo_data_subpath, resolve_repo_data_path, load_perception_records, dataset_provenance
from blox.env import Block
from blox.constants import Orientation, TileType
from blox.perception_helpers import board_state_prompt, extract_board_state, normalize_height_map
from llm_backends import make_backend
from eval_utils import positive_int, nonnegative_int
from eval_utils import encode_pil_image, image_content_block, strip_images_for_log, aggregate_usage


# -- Tile type int -> symbol mapping --
TILE_SYMBOL = {
    int(TileType.VOID): "#",
    int(TileType.NORMAL): ".",
    int(TileType.GOAL): "G",
    int(TileType.FRAGILE): "F",
    int(TileType.CRUMBLE): "C",
    int(TileType.SWITCH): "S",
    int(TileType.BRIDGE): "~",  # default OFF representation
    int(TileType.ELEVATOR): "E",
}

ORIENT_NAME = {
    int(Orientation.STANDING): "STANDING",
    int(Orientation.LYING_H): "LYING_H",
    int(Orientation.LYING_V): "LYING_V",
}


def build_messages(prompt: str, image: Image.Image) -> List[Dict[str, Any]]:
    base64_image = encode_pil_image(image)
    img_block = image_content_block(base64_image)
    return [{"role": "user", "content": [img_block, {"type": "text", "text": prompt}]}]


def load_local_dataset(load_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load frozen images and compute labels from their frozen board/block states."""
    dataset = []
    for level_data in load_perception_records(load_dir, max_samples):
        level_id = level_data["id"]
        image = level_data["image"]

        # Ground truth
        board_data = level_data["board"]
        rows, cols = board_data["rows"], board_data["cols"]
        gt_grid = [[TILE_SYMBOL.get(cell, "#") for cell in row] for row in board_data["grid"]]

        block_data = level_data["block"]
        orient = Orientation(block_data["orientation"])
        gt_block_cells = set(Block(block_data["r"], block_data["c"], orient).cells())
        gt_block_orient = ORIENT_NAME[block_data["orientation"]]

        goal_data = level_data["goal"]
        gt_goal_pos = (goal_data["r"], goal_data["c"])

        gt_crumble_cells = set()
        for r_idx, row in enumerate(board_data["grid"]):
            for c_idx, cell in enumerate(row):
                if cell == int(TileType.CRUMBLE):
                    gt_crumble_cells.add((r_idx, c_idx))

        # Elevator ground truth
        gt_elevator_cells = set()
        for r_idx, row in enumerate(board_data["grid"]):
            for c_idx, cell in enumerate(row):
                if cell == int(TileType.ELEVATOR):
                    gt_elevator_cells.add((r_idx, c_idx))

        # Elevated cells and height map ground truth
        gt_height_map = None
        gt_elevated_cells = set()
        if "height_map" in board_data:
            hm = board_data["height_map"]
            # Void cells have no visible surface; the answer contract gives them height 0.
            gt_height_map = normalize_height_map(board_data["grid"], hm)
            for r_idx, row in enumerate(hm):
                for c_idx, val in enumerate(row):
                    if int(val) == 1 and board_data["grid"][r_idx][c_idx] != int(TileType.VOID):
                        gt_elevated_cells.add((r_idx, c_idx))

        # Block height
        gt_block_height = level_data.get("current_height", 0)

        dataset.append({
            "image": image,
            "rows": rows,
            "cols": cols,
            "gt_grid": gt_grid,
            "gt_block_cells": gt_block_cells,
            "gt_block_orient": gt_block_orient,
            "gt_goal_pos": gt_goal_pos,
            "gt_crumble_cells": gt_crumble_cells,
            "gt_elevator_cells": gt_elevator_cells,
            "gt_elevated_cells": gt_elevated_cells,
            "gt_height_map": gt_height_map,
            "gt_block_height": gt_block_height,
            "_level_file": level_id,
        })

    return dataset


def compute_metrics(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-sample metrics comparing ground truth to prediction."""
    metrics: Dict[str, Any] = {}

    # Grid accuracy
    gt_grid = np.array(gt["gt_grid"], dtype=str)
    pred_grid = pred.get("grid")
    if pred_grid is not None and pred_grid.shape == gt_grid.shape:
        cell_matches = (gt_grid == pred_grid)
        metrics["grid_cell_accuracy"] = float(cell_matches.mean())
        metrics["whole_grid_accuracy"] = float(cell_matches.all())
    else:
        metrics["grid_cell_accuracy"] = 0.0
        metrics["whole_grid_accuracy"] = False

    # Block cells (set comparison — order doesn't matter)
    pred_cells = pred.get("block_cells")
    gt_cells = gt["gt_block_cells"]
    metrics["block_pos_accuracy"] = float(
        pred_cells is not None and pred_cells == gt_cells
    )

    # Block orientation
    metrics["block_orient_accuracy"] = float(
        str(pred.get("block_orient", "")).upper() == gt["gt_block_orient"]
    )

    # Goal position
    metrics["goal_accuracy"] = float(pred.get("goal_pos") == gt["gt_goal_pos"])

    # Crumble cells accuracy (set comparison)
    gt_crumble = gt.get("gt_crumble_cells", set())
    if gt_crumble:
        pred_crumble = set()
        if pred_grid is not None:
            for r in range(pred_grid.shape[0]):
                for c in range(pred_grid.shape[1]):
                    if pred_grid[r, c] == "C":
                        pred_crumble.add((r, c))
        metrics["crumble_accuracy"] = float(pred_crumble == gt_crumble)
    else:
        metrics["crumble_accuracy"] = None

    # Elevator cells accuracy (set comparison)
    gt_elevator = gt.get("gt_elevator_cells", set())
    if gt_elevator:
        # Extract elevator cells from predicted grid
        pred_elevator = set()
        if pred_grid is not None:
            for r in range(pred_grid.shape[0]):
                for c in range(pred_grid.shape[1]):
                    if pred_grid[r, c] == "E":
                        pred_elevator.add((r, c))
        metrics["elevator_accuracy"] = float(pred_elevator == gt_elevator)
    else:
        metrics["elevator_accuracy"] = None  # no elevators in this level

    # Elevated cells accuracy (set comparison)
    gt_elevated = gt.get("gt_elevated_cells", set())
    if gt_elevated:
        pred_height_map = pred.get("height_map")
        pred_elevated = set()
        if isinstance(pred_height_map, np.ndarray):
            for r in range(pred_height_map.shape[0]):
                for c in range(pred_height_map.shape[1]):
                    if int(pred_height_map[r, c]) == 1:
                        pred_elevated.add((r, c))
        metrics["elevated_accuracy"] = float(pred_elevated == gt_elevated)
    else:
        metrics["elevated_accuracy"] = None  # no elevated cells

    # Block height accuracy
    gt_bh = gt.get("gt_block_height", 0)
    pred_bh = pred.get("block_height", 0)
    metrics["block_height_accuracy"] = float(pred_bh == gt_bh)

    # Height map accuracy (cell-level)
    gt_hm = gt.get("gt_height_map")
    if gt_hm is not None:
        gt_hm_arr = np.array(gt_hm, dtype=int)
        pred_hm = pred.get("height_map")
        if isinstance(pred_hm, np.ndarray) and pred_hm.shape == gt_hm_arr.shape:
            metrics["height_map_accuracy"] = float((pred_hm == gt_hm_arr).mean())
        else:
            metrics["height_map_accuracy"] = 0.0
    else:
        metrics["height_map_accuracy"] = None

    # Whole state (include elevator/elevated/height when present)
    whole = (
        metrics["whole_grid_accuracy"]
        and metrics["block_pos_accuracy"]
        and metrics["block_orient_accuracy"]
        and metrics["goal_accuracy"]
        and metrics["block_height_accuracy"]
    )
    if metrics["elevator_accuracy"] is not None:
        whole = whole and metrics["elevator_accuracy"]
    if metrics["elevated_accuracy"] is not None:
        whole = whole and metrics["elevated_accuracy"]
    metrics["whole_state_accuracy"] = float(whole)

    return metrics


def process_sample(
    i: int,
    sample: Dict[str, Any],
    model_name: str,
    backend_inst: Any = None,
    local_server_port: Optional[int] = None,
    local_server_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: int = 32768,
) -> Dict[str, Any]:
    """Process a single sample: call LLM, parse response, compute metrics."""
    be = backend_inst or make_backend(model_name, local_server_port=local_server_port, local_server_url=local_server_url)
    prompt = board_state_prompt(sample["rows"], sample["cols"])
    messages = build_messages(prompt, sample["image"])
    chat_result = be.chat(messages, max_tokens=max_tokens, model=model_name, reasoning_effort=reasoning_effort)
    response = chat_result.content
    reasoning = chat_result.reasoning_content or None
    usage = chat_result.usage

    # Parse response
    pred: Dict[str, Any] = {}
    parse_error = None
    try:
        pred = extract_board_state(response, sample["rows"], sample["cols"])
    except Exception as e:
        parse_error = str(e)

    metrics = compute_metrics(sample, pred)

    level_name = public_data_path(sample["_level_file"])
    crumble_str = f" crumble={metrics['crumble_accuracy']:.0f}" if metrics.get('crumble_accuracy') is not None else ""
    elev_str = f" elev={metrics['elevator_accuracy']:.0f}" if metrics.get('elevator_accuracy') is not None else ""
    elevated_str = f" elevated={metrics['elevated_accuracy']:.0f}" if metrics.get('elevated_accuracy') is not None else ""
    height_str = f" bh={metrics['block_height_accuracy']:.0f}"
    tqdm.write(
        f"Sample {i} | {level_name} | "
        f"grid_cell={metrics['grid_cell_accuracy']:.2f} "
        f"block_pos={metrics['block_pos_accuracy']:.0f} "
        f"orient={metrics['block_orient_accuracy']:.0f} "
        f"goal={metrics['goal_accuracy']:.0f}"
        f"{crumble_str}{elev_str}{elevated_str}{height_str}"
    )

    result = {
        "id": sample["_level_file"],
        "level_id": public_data_path(sample["_level_file"]),
        "messages": strip_images_for_log(messages),
        "response": response,
        "ground_truth": {
            "grid": sample["gt_grid"],
            "block_cells": sorted(sample["gt_block_cells"]),
            "block_orient": sample["gt_block_orient"],
            "goal_pos": list(sample["gt_goal_pos"]),
            "crumble_cells": sorted(sample["gt_crumble_cells"]),
            "elevator_cells": sorted(sample["gt_elevator_cells"]),
            "elevated_cells": sorted(sample["gt_elevated_cells"]),
            "block_height": sample["gt_block_height"],
            "height_map": sample["gt_height_map"],
        },
        "prediction": {
            "grid": pred["grid"].tolist() if isinstance(pred.get("grid"), np.ndarray) else pred.get("grid"),
            "block_cells": sorted(pred["block_cells"]) if pred.get("block_cells") else None,
            "block_orient": pred.get("block_orient"),
            "goal_pos": list(pred["goal_pos"]) if pred.get("goal_pos") else None,
            "block_height": pred.get("block_height"),
            "height_map": pred["height_map"].tolist() if isinstance(pred.get("height_map"), np.ndarray) else pred.get("height_map"),
        },
        "parse_error": parse_error,
        **metrics,
        "reasoning_content": reasoning,
        "usage": usage,
    }
    return result


def aggregate(log_path: str) -> Dict[str, Any]:
    """Aggregate per-sample results into summary metrics."""
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return {}

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return {}

    metric_keys = [
        "grid_cell_accuracy", "whole_grid_accuracy",
        "block_pos_accuracy", "block_orient_accuracy",
        "goal_accuracy", "crumble_accuracy", "elevator_accuracy", "elevated_accuracy",
        "block_height_accuracy", "height_map_accuracy",
        "whole_state_accuracy",
    ]
    summary: Dict[str, Any] = {"n_samples": len(records)}
    for key in metric_keys:
        vals = [r[key] for r in records if key in r and r[key] is not None]
        summary[key] = float(np.mean(vals)) if vals else None

    return summary


def evaluate(args):
    # Output paths
    model_safe = args.model_name.replace("/", "-").replace("\\", "-")
    save_dir = os.path.join(args.output_dir, "blox", "perception", model_safe, "heldout")
    os.makedirs(save_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"results__{run_id}.jsonl")

    # Load data
    dataset = load_local_dataset(args.load_dir, args.max_samples)
    view_label = "isometric 3D"
    print(f"Loaded {len(dataset)} samples from {public_data_path(args.load_dir)} ({view_label})")

    if not dataset:
        print("No samples found. Check --load_dir.")
        return

    backend = make_backend(args.model_name, local_server_port=args.local_server_port, local_server_url=args.local_server_url)

    # Resume: find already-completed sample IDs from previous runs
    completed_ids = set()
    for prev_file in glob.glob(os.path.join(save_dir, "results__*.jsonl")):
        with open(prev_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get('id'))
                except Exception:
                    pass
    indexed_dataset = [(i, s) for i, s in enumerate(dataset) if s["_level_file"] not in completed_ids]
    if len(indexed_dataset) < len(dataset):
        print(f"Resuming: {len(completed_ids)} done, {len(indexed_dataset)}/{len(dataset)} remaining")
    if not indexed_dataset:
        print("All samples already completed.")

    if args.num_workers <= 1:
        for i, sample in tqdm(indexed_dataset, total=len(indexed_dataset)):
            result = process_sample(i, sample, args.model_name, backend, local_server_port=args.local_server_port, local_server_url=args.local_server_url, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_sample, i, sample, args.model_name, None, local_server_port=args.local_server_port, local_server_url=args.local_server_url, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
                for i, sample in indexed_dataset
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                except Exception as e:
                    tqdm.write(f"Worker failed: {e}")
                    continue
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Merge all result files for aggregation (supports resume)
    all_records = {}
    for rf in sorted(glob.glob(os.path.join(save_dir, "results__*.jsonl"))):
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    legacy_key = "level_" + "path"
                    if legacy_key in rec:
                        rec["level_id"] = public_data_path(rec.pop(legacy_key))
                    all_records[rec['id']] = rec
                except Exception:
                    pass
    merged_file = os.path.join(save_dir, "results__merged.jsonl")
    with open(merged_file, 'w', encoding='utf-8') as f:
        for rec in sorted(all_records.values(), key=lambda r: r['id']):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = aggregate(merged_file)
    usage_list = [rec.get('usage') for rec in all_records.values()]

    summary.update({
        "model": args.model_name,
        "view": "isometric",
        "dataset": dataset_provenance(args.load_dir),
        "data_dir": public_data_path(args.load_dir),
        "max_samples": args.max_samples,
        "run_id": run_id,
        "run_log": os.path.basename(log_file),
        "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
        "total_usage": aggregate_usage(usage_list),
    })

    final_file = os.path.join(save_dir, f"summary_final_{run_id}.jsonl")
    with open(final_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Samples: {summary.get('n_samples', 0)}")
    for key in ["grid_cell_accuracy", "whole_grid_accuracy", "block_pos_accuracy",
                 "block_orient_accuracy", "goal_accuracy", "crumble_accuracy", "elevator_accuracy",
                 "elevated_accuracy", "block_height_accuracy", "height_map_accuracy",
                 "whole_state_accuracy"]:
        val = summary.get(key)
        if val is not None:
            print(f"  {key}: {val:.4f}")
    print(f"Log: {log_file}")
    print(f"Summary: {final_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blox perception evaluation: board state parsing")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--load_dir", type=str, default=repo_data_subpath("blox", "perception"),
                        help="Root dir or JSONL file containing released Blox perception levels")
    parser.add_argument("--max_samples", type=nonnegative_int, default=None,
                        help="Max total samples to evaluate")
    parser.add_argument("--num_workers", type=positive_int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--local_server_port", type=int, default=None,
                        help="Port of a running local inference server (vLLM or SGLang) to use (e.g. 8000)")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        help="Reasoning effort level (for thinking models)")
    parser.add_argument("--local_server_url", type=str, default=None,
                        help="OpenAI-compatible base URL; overrides --local_server_port.")
    parser.add_argument("--max_tokens", type=positive_int, default=32768,
                        help="Maximum completion tokens per model call.")
    parser.add_argument("--output_dir", default="eval/output", help="Evaluation output root")
    args = parser.parse_args()
    args.load_dir = str(resolve_repo_data_path(args.load_dir))
    # Backward compatibility alias
    args.model_name = args.model

    evaluate(args)
