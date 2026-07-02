from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_from_space.pipeline.common import initialize_run
from retrieval_from_space.pipeline.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions for a run.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML pipeline config.")
    parser.add_argument("--run-id", required=True, help="Run id under output_root.")
    args = parser.parse_args()

    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    evaluate(config, paths, state)
    print(paths.root)


if __name__ == "__main__":
    main()
