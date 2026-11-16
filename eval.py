"""Single entry point for MARBLE model evaluation."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


TASK_MODULES = {
    ("cube", "planning"): "cube.eval_planning",
    ("cube", "perception"): "cube.eval_perception",
    ("maze", "planning"): "maze.eval_planning",
    ("maze", "perception"): "maze.eval_perception",
    ("blox", "planning"): "blox.eval_planning",
    ("blox", "perception"): "blox.eval_perception",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MARBLE evaluations through one shared CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--task", choices=["cube", "maze", "blox"], required=True)
    parser.add_argument("--mode", choices=["planning", "perception"], required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Use local data under this root instead of Hugging Face (cube/, maze/, blox/).",
    )
    # Once a task/mode is selected, its evaluator supplies the detailed help.
    if any(flag in sys.argv for flag in ("-h", "--help")) and not all(
        flag in sys.argv for flag in ("--task", "--mode")
    ):
        parser.print_help()
        parser.exit()
    args, forwarded = parser.parse_known_args()
    return args, forwarded


def main() -> None:
    args, forwarded = parse_args()
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if args.data_root:
        os.environ["MARBLE_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    module = TASK_MODULES[(args.task, args.mode)]
    sys.argv = [module, *forwarded]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
