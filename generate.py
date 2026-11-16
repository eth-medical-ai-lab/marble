"""Generate MARBLE planning datasets through one shared command."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cube", "maze", "blox"], required=True)
    parser.add_argument("--config", choices=["easy", "hard"], required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True, help="Output data root.")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num_samples must be positive")

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module(f"{args.task}.generator.planning")
    destination = args.output_dir.expanduser().resolve() / args.task / args.config
    if destination.exists() and any(destination.iterdir()):
        parser.error(f"Output split is not empty: {destination}. Use a new output directory.")
    destination.mkdir(parents=True, exist_ok=True)

    # Publish samples.jsonl only after every requested record has been generated.
    # Interrupted runs retain the marker and partial file and cannot be silently reused.
    marker = destination / ".incomplete"
    marker.touch()
    options = {"num_samples": args.num_samples, "seed": args.seed,
               "output_dir": destination, "config": args.config}
    seeds = []
    partial = destination / "samples.jsonl.partial"
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for record in module.generate(**options):
            handle.write(json.dumps(record) + "\n")
            seeds.append(record["seed"])
            print(f"Generated {len(seeds)}/{args.num_samples}", flush=True)
    if len(seeds) != args.num_samples:
        raise RuntimeError(f"Generated {len(seeds)}/{args.num_samples}; output is incomplete")

    # Record the exact source and output hashes.
    files = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(destination.rglob("*")) if path.is_file() and path != marker}
    files["samples.jsonl"] = files.pop("samples.jsonl.partial")
    source_files = [root / "generate.py", *sorted((root / "src").rglob("*.py"))]
    manifest = {
        "task": args.task, "mode": "planning", "config": args.config, "num_samples": args.num_samples,
        "seed": args.seed, "accepted_seeds": seeds, "recipe": module.RECIPE,
        "files": files,
        "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.rename(destination / "samples.jsonl")
    marker.unlink()
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
