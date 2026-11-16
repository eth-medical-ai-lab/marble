import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

from PIL import Image
import numpy as np
import glob
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_paths import repo_data_subpath, resolve_repo_data_path, load_perception_records, dataset_provenance
from llm_backends import make_backend
from eval_utils import positive_int, nonnegative_int
from eval_utils import encode_pil_image, image_content_block, strip_images_for_log, create_text_box, aggregate_usage
from maze.perception_helpers import board_parsing_prompt, extract_2d_array


def build_messages(prompt: str, image: Image.Image) -> List[Dict[str, Any]]:
    base64_image = encode_pil_image(image)
    img_block = image_content_block(base64_image)
    return [{
        "role": "user",
        "content": [img_block, {"type": "text", "text": prompt}],
    }]


def load_local_dataset(load_dir: str, max_samples: int | None = None) -> List[Dict[str, Any]]:
    """Load the single fixed 7x7 board-parsing set."""
    records = load_perception_records(load_dir, max_samples)
    for record in records:
        record["answer"] = ast.literal_eval(record["grid_tokens_str"])
        record["question"] = board_parsing_prompt(7)
    return records


def evaluate(args):
    model_safe = args.model.replace('/', '-').replace('\\', '-')
    save_path = os.path.join(args.output_dir, 'maze', 'perception', model_safe, 'heldout')
    os.makedirs(save_path, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_path, f"{model_safe}__{run_id}.jsonl")
    
    if not os.path.exists(log_file):
        open(log_file, 'a', encoding='utf-8').close()

    dataset = load_local_dataset(args.load_dir, args.max_samples)

    backend = make_backend(args.model, local_server_port=args.local_server_port, local_server_url=args.local_server_url)

    def process_sample(i: int, sample: Dict[str, Any], model_name: str, backend_inst: Any | None = None) -> Dict[str, Any]:
        be = backend_inst or make_backend(model_name, local_server_port=args.local_server_port, local_server_url=args.local_server_url)
        prompt = sample['question']
        image = sample['image']
        messages = build_messages(prompt, image)
        chat_result = be.chat(messages, max_tokens=args.max_tokens, model=model_name, reasoning_effort=getattr(args, 'reasoning_effort', None))
        response = chat_result.content
        reasoning = chat_result.reasoning_content or None
        usage = chat_result.usage

        tqdm.write(f"\nSample {i} • task=board_parsing • model={model_name}")
        tqdm.write("   📤 Response:")
        for line in create_text_box(response or "", box_width=100, prefix="   "):
            tqdm.write(line)
        if reasoning:
            tqdm.write("   🧠 Reasoning:")
            for line in create_text_box(str(reasoning) or "", box_width=100, prefix="   "):
                tqdm.write(line)

        try:
            solution = extract_2d_array(response)
        except Exception as e:
            print(f"Error extracting 2D array: {e}")
            solution = None

        gt_answer = np.array(sample['answer'])

        if solution is not None:
            same_shape = gt_answer.shape == solution.shape
            if same_shape:
                gt_flat = gt_answer.flatten()
                pred_flat = solution.flatten()

                def tokens_equal(gt_token: str, pred_token: str) -> bool:
                    if gt_token == pred_token:
                        return True
                    if len(gt_token) >= 2 and len(pred_token) >= 2 and gt_token[0] == 'I' and pred_token[0] == 'I':
                        return (int(gt_token[1]) % 2) == (int(pred_token[1]) % 2)
                    return False

                matches = [tokens_equal(gt, pr) for gt, pr in zip(gt_flat, pred_flat)]
                per_cell_acc = float(np.mean(matches))
                whole_board_acc = bool(all(matches))

                def rotation_match(gt_token, pred_token):
                    if len(gt_token) < 2 or len(pred_token) < 2:
                        return False
                    if gt_token[1:] == pred_token[1:]:
                        return True
                    if gt_token[0] == 'I':
                        gt_rot, pred_rot = int(gt_token[1]), int(pred_token[1])
                        return (gt_rot % 2) == (pred_rot % 2)
                    return False

                tile_type_acc = float(np.mean([gt[0] == pred[0] for gt, pred in zip(gt_flat, pred_flat) if len(gt) > 0 and len(pred) > 0]))
                rotation_acc = float(np.mean([rotation_match(gt, pred) for gt, pred in zip(gt_flat, pred_flat)]))
                tile_breakdown = {}

                for tile_type in ['I', 'L', 'T']:
                    gt_type = [gt for gt in gt_flat if len(gt) > 0 and gt[0] == tile_type]
                    pred_type = [pred for gt, pred in zip(gt_flat, pred_flat) if len(gt) > 0 and gt[0] == tile_type and len(pred) > 0]
                    if gt_type and pred_type:
                        type_acc = float(np.mean([gt[0] == pred[0] for gt, pred in zip(gt_type, pred_type)]))
                        rot_acc = float(np.mean([rotation_match(gt, pred) for gt, pred in zip(gt_type, pred_type)]))
                        tile_breakdown[f'{tile_type}_type_acc'] = type_acc
                        tile_breakdown[f'{tile_type}_rotation_acc'] = rot_acc
            else:
                per_cell_acc, whole_board_acc = 0.0, False
                tile_type_acc, rotation_acc = 0.0, 0.0
                tile_breakdown = {}

        else:
            per_cell_acc, whole_board_acc = 0.0, False
            tile_type_acc, rotation_acc = 0.0, 0.0
            tile_breakdown = {}

        result = {
            'id': sample['id'],
            'messages': strip_images_for_log(messages),
            'answer': sample['answer'],
            'response': response,
            'per_cell_acc': per_cell_acc,
            'whole_board_acc': whole_board_acc,
            'tile_type_acc': tile_type_acc,
            'rotation_acc': rotation_acc,
            **tile_breakdown,
            'image_name': os.path.basename(sample.get('image_path', '')),
            'annotation_name': sample['id'],
            'reasoning_content': reasoning,
            'usage': usage,
        }


        return result

    # Resume: find already-completed sample IDs from previous runs
    completed_ids = set()
    for prev_file in glob.glob(os.path.join(save_path, f"{model_safe}__*.jsonl")):
        with open(prev_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get('id'))
                except Exception:
                    pass
    indexed_dataset = [(i, s) for i, s in enumerate(dataset) if s['id'] not in completed_ids]
    if len(indexed_dataset) < len(dataset):
        print(f"Resuming: {len(completed_ids)} done, {len(indexed_dataset)}/{len(dataset)} remaining")
    if not indexed_dataset:
        print("All samples already completed.")

    success_count = 0

    if args.num_workers <= 1:
        for i, sample in tqdm(indexed_dataset, total=len(indexed_dataset)):
            result = process_sample(i, sample, args.model, backend)
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
            success_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_sample, i, sample, args.model, None) for i, sample in indexed_dataset]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                except Exception as e:
                    tqdm.write(f"Worker failed: {e}")
                    continue
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                success_count += 1

    def aggregate(log_path: str) -> Dict[str, Any]:
        with open(log_path, 'r', encoding='utf-8') as f:
            p, w = [], []
            ttype, trot = [], []
            tbd: Dict[str, List[Any]] = {f'{t}_{m}_acc': [] for t in ['I','L','T'] for m in ['type','rotation']}

            for l in f:
                try:
                    rec = json.loads(l)
                    p.append(rec.get('per_cell_acc'))
                    w.append(rec.get('whole_board_acc'))
                    ttype.append(rec.get('tile_type_acc'))
                    trot.append(rec.get('rotation_acc'))
                    for key in tbd:
                        tbd[key].append(rec.get(key))
                except Exception:
                    continue

            result = {
                'per_cell_accuracy': float(np.mean([x for x in p if x is not None])) if p else None,
                'whole_board_accuracy': float(np.mean([x for x in w if x is not None])) if w else None,
            }

            if ttype:
                result['tile_type_accuracy'] = float(np.mean([x for x in ttype if x is not None]))

            if trot:
                result['rotation_accuracy'] = float(np.mean([x for x in trot if x is not None]))

            for key, values in tbd.items():
                if values and any(x is not None for x in values):
                    result[key.replace('_acc', '_accuracy')] = float(np.mean([x for x in values if x is not None]))

            return result


    # Merge all result files for aggregation (supports resume)
    all_records = {}
    for rf in sorted(glob.glob(os.path.join(save_path, f"{model_safe}__*.jsonl"))):
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    all_records[rec['id']] = rec
                except Exception:
                    pass
    merged_file = os.path.join(save_path, f"{model_safe}__merged.jsonl")
    with open(merged_file, 'w', encoding='utf-8') as f:
        for rec in sorted(all_records.values(), key=lambda r: r['id']):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    result = aggregate(merged_file)
    usage_list = [rec.get('usage') for rec in all_records.values()]

    final_log_file = os.path.join(save_path, f"{model_safe}_final_{run_id}.jsonl")

    with open(final_log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'model': args.model,
            'dataset': dataset_provenance(args.load_dir),
            'run_id': run_id,
            'run_log': os.path.basename(log_file),
            'timestamp': datetime.now().isoformat(timespec='seconds') + 'Z',
            'total_usage': aggregate_usage(usage_list),
            **result
        }) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--load_dir', type=str, default=repo_data_subpath("maze", "perception"), help='Held-out directory with samples.jsonl and images/')
    parser.add_argument('--max_samples', type=nonnegative_int, default=None, help='max samples to evaluate')
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

    evaluate(args)
