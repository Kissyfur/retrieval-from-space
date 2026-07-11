from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import initialize_run
from src.pipeline.run import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full retrieval-from-space pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML pipeline config.")
    parser.add_argument("--run-id", help="Existing or explicit run id.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw/matchup artifacts.")
    parser.add_argument(
        "--ask-problem-type",
        action="store_true",
        help="Ask whether the problem is classification or regression if missing from config.",
    )
    args = parser.parse_args()

    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    run_pipeline(config, paths, state, overwrite=args.overwrite, interactive=args.ask_problem_type)
    print(paths.root)


if __name__ == "__main__":
    main()
