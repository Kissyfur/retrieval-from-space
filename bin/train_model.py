from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_from_space.pipeline.common import initialize_run
from retrieval_from_space.pipeline.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model from prepared datasets.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML pipeline config.")
    parser.add_argument("--run-id", help="Existing or explicit run id.")
    parser.add_argument(
        "--stage",
        choices=["all", "base", "final"],
        default="all",
        help="Training stage to run. Use 'base' to train base models only, or 'final' to train from saved base signals.",
    )
    parser.add_argument(
        "--base-name",
        action="append",
        dest="base_names",
        help="Base model name to train when --stage base is used. Repeat for multiple names.",
    )
    parser.add_argument(
        "--ask-problem-type",
        action="store_true",
        help="Ask whether the problem is classification or regression if missing from config.",
    )
    args = parser.parse_args()

    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    train(
        config,
        paths,
        state,
        interactive=args.ask_problem_type,
        stage=args.stage,
        base_names=args.base_names,
    )
    print(paths.root)


if __name__ == "__main__":
    main()
