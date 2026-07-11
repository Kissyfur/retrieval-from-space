from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.pipeline.common import initialize_run
from src.pipeline.download import download_products


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare configured products for matchup creation.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML pipeline config.")
    parser.add_argument("--run-id", help="Existing or explicit run id.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local raw products or remote product markers.",
    )
    args = parser.parse_args()

    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    download_products(config, paths, state, overwrite=args.overwrite)
    print(paths.root)


if __name__ == "__main__":
    main()
