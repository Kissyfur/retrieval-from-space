from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.examples.synthetic import create_synthetic_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local synthetic inputs for an end-to-end smoke run.")
    parser.add_argument("--output-dir", default="examples/synthetic/generated")
    parser.add_argument("--n-observations", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts = create_synthetic_example(args.output_dir, args.n_observations, args.seed)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
