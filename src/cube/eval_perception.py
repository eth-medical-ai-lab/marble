"""Evaluate the held-out CUBE perception dataset."""
from __future__ import annotations
import argparse
import ast
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from data_paths import public_data_path, repo_data_subpath, resolve_repo_data_path, load_perception_records, dataset_provenance
from llm_backends import make_backend, LLMBackend
from eval_utils import positive_int, nonnegative_int
from eval_utils import encode_pil_image, image_content_block, strip_images_for_log, aggregate_usage


def extract_2d_array(response: str):
    """Read the requested 5x5 binary array, allowing the previous answer prefix."""
    match = re.search(r"\[\s*\[", response)
    if match is None:
        raise ValueError("No array in response")
    depth = 0
    for i in range(match.start(), len(response)):
        depth += (response[i] == "[") - (response[i] == "]")
        if depth == 0:
            array = np.array(ast.literal_eval(response[match.start():i + 1]))
            if array.shape != (5, 5) or not np.isin(array, [0, 1]).all():
                raise ValueError("Expected a 5x5 binary array")
            return array
    raise ValueError("Unbalanced array")


load_dotenv()


DEFAULT_QUESTION = (
    "Extract the 5x5 binary grid for this Happy-Cube piece. "
    "Use Edge 1 as left, Edge 2 as top, Edge 3 as right, and Edge 4 as bottom. "
    "Rows run top to bottom and columns left to right. "
    "Return only a Python-style 5x5 list of 0/1 integers (0 = gap, 1 = solid)."
)


def _base_dir(args) -> str:
    model_dir = args.model.replace("/", "-").replace("\\", "-")
    return os.path.join(args.output_dir, "cube", "perception", model_dir, "heldout")


def load_local_dataset(load_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    records = load_perception_records(load_dir, max_samples)
    for record in records:
        record["question"] = DEFAULT_QUESTION
    return records


def build_messages(sample: Dict) -> List[Dict]:
    """Build chat messages with piece image + question prompt."""
    prompt = sample['question']

    b64 = encode_pil_image(sample['image'])
    img_block = image_content_block(b64)

    return [{"role": "user", "content": [img_block, {"type": "text", "text": prompt}]}]


def process_sample(
    index: int,
    sample: Dict,
    model: str,
    backend: LLMBackend,
    reasoning_effort: str = None,
    max_tokens: int = 32768,
) -> Dict[str, Any]:
    """Evaluate a single perception sample."""
    messages = build_messages(sample)
    chat_result = backend.chat(messages, model=model, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    response = chat_result.content
    reasoning_content = chat_result.reasoning_content or ""
    usage = chat_result.usage

    try:
        solution = extract_2d_array(response)
    except (ValueError, SyntaxError, TypeError):
        solution = None

    gt_answer = np.array(sample['answer'])
    if solution is not None:
        per_cell_acc = float(np.equal(solution, gt_answer).mean())
        whole_piece_acc = bool(np.all(np.equal(solution, gt_answer)))
    else:
        per_cell_acc, whole_piece_acc = 0.0, False

    return {
        'id': sample['id'],
        'messages': strip_images_for_log(messages),
        'answer': sample['answer'],
        'response': response,
        'per_cell_acc': per_cell_acc,
        'whole_piece_acc': whole_piece_acc,
        'reasoning_content': reasoning_content,
        'usage': usage,
    }


def aggregate(log_path: str) -> Dict[str, Any]:
    """Aggregate JSONL results into summary metrics."""
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return {}

    per_cell, whole_piece = [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                per_cell.append(rec['per_cell_acc'])
                whole_piece.append(rec['whole_piece_acc'])
            except Exception:
                continue

    return {
        'n_samples': len(per_cell),
        'per_cell_accuracy': float(np.mean(per_cell)) if per_cell else None,
        'whole_piece_accuracy': float(np.mean(whole_piece)) if whole_piece else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Cube perception evaluation: piece grid extraction")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_samples', type=nonnegative_int, default=None, help='Max samples to evaluate')
    parser.add_argument('--load_dir', type=str, default=repo_data_subpath("cube", "perception"),
                        help='Local Cube perception data directory or JSONL file')
    parser.add_argument('--num_workers', type=positive_int, default=1, help='Number of parallel workers (1 = sequential)')
    parser.add_argument('--local_server_port', type=int, default=None, help='Port of a running local inference server (vLLM or SGLang) to use (e.g. 8000)')
    parser.add_argument('--reasoning_effort', type=str, default=None, help='Reasoning effort level (for thinking models)')
    parser.add_argument("--local_server_url", type=str, default=None,
                        help="OpenAI-compatible base URL; overrides --local_server_port.")
    parser.add_argument("--max_tokens", type=positive_int, default=32768,
                        help="Maximum completion tokens per model call.")
    parser.add_argument("--output_dir", default="eval/output", help="Evaluation output root")
    args = parser.parse_args()


    args.load_dir = str(resolve_repo_data_path(args.load_dir))
    dataset = load_local_dataset(args.load_dir, args.max_samples)
    print(f"Loaded {len(dataset)} local Cube perception samples from {public_data_path(args.load_dir)}")

    # Output paths
    base_dir = _base_dir(args)
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_safe = args.model.replace('/', '-').replace('\\', '-')
    log_file = os.path.join(base_dir, f"results__{run_id}.jsonl")

    # Resume: find already-completed sample IDs from previous runs
    completed_ids = set()
    for prev_file in glob.glob(os.path.join(base_dir, f"results__*.jsonl")):
        with open(prev_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get('id'))
                except Exception:
                    pass
    # Build (index, sample) pairs, skipping completed
    indexed_dataset = [(i, s) for i, s in enumerate(dataset) if s['id'] not in completed_ids]
    if len(indexed_dataset) < len(dataset):
        print(f"Resuming: {len(completed_ids)} done, {len(indexed_dataset)}/{len(dataset)} remaining")
    if not indexed_dataset:
        print("All samples already completed.")

    backend = make_backend(args.model, local_server_port=args.local_server_port, local_server_url=args.local_server_url)
    success_count = 0

    if args.num_workers <= 1:
        for i, sample in tqdm(indexed_dataset, total=len(indexed_dataset)):
            result = process_sample(i, sample, args.model, backend,
                                    reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            success_count += 1

            print(f"  [{i}] per_cell={result['per_cell_acc']:.2f} whole_piece={result['whole_piece_acc']}")
    else:
        import itertools
        worker_backends = [make_backend(args.model, local_server_port=args.local_server_port, local_server_url=args.local_server_url)
                           for _ in range(args.num_workers)]
        backend_cycle = itertools.cycle(worker_backends)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_sample, i, sample, args.model,
                                next(backend_cycle),
                                reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
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
                success_count += 1

    # Aggregate across ALL result files (supports resume)
    all_results_files = sorted(glob.glob(os.path.join(base_dir, f"results__*.jsonl")))
    all_records = {}
    for rf in all_results_files:
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    all_records[rec['id']] = rec  # dedup by sample ID
                except Exception:
                    pass
    # Write merged results for aggregation
    merged_file = os.path.join(base_dir, f"results__merged.jsonl")
    with open(merged_file, 'w', encoding='utf-8') as f:
        for rec in sorted(all_records.values(), key=lambda r: r['id']):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = aggregate(merged_file)
    usage_list = [rec.get('usage') for rec in all_records.values()]

    summary.update({
        'model': args.model,
        'view': 'isometric',
        'dataset': dataset_provenance(args.load_dir),
        'data_dir': public_data_path(args.load_dir),
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds') + 'Z',
        'total_usage': aggregate_usage(usage_list),
    })

    final_file = os.path.join(base_dir, f"results_final_{run_id}.jsonl")
    with open(final_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nResults: per_cell={summary.get('per_cell_accuracy', 0):.4f} "
          f"whole_piece={summary.get('whole_piece_accuracy', 0):.4f}")
    print(f"Log: {log_file}")
    print(f"Summary: {final_file}")


if __name__ == '__main__':
    main()
