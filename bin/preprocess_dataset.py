from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import initialize_run
from src.pipeline.preprocess import preprocess_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess matchups into train-ready datasets.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML pipeline config.")
    parser.add_argument("--run-id", help="Existing or explicit run id.")
    args = parser.parse_args()

    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    preprocess_datasets(config, paths, state)
    print(paths.root)


if __name__ == "__main__":
    main()
