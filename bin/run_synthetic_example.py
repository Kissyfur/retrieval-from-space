from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.examples.synthetic import create_synthetic_example
from src.pipeline.common import initialize_run
from src.pipeline.run import run_pipeline


def _assert_outputs(run_root: Path) -> None:
    expected = [
        run_root / "run_manifest.json",
        run_root / "config" / "config.json",
        run_root / "pipeline_state.json",
        run_root / "raw" / "synthetic_reflectance.nc",
        run_root / "raw" / "synthetic_physics.nc",
        run_root / "processed" / "targets.csv",
        run_root / "processed" / "matchups" / "synthetic_reflectance.nc",
        run_root / "processed" / "matchups" / "synthetic_physics.nc",
        run_root / "datasets" / "target.nc",
        run_root / "datasets" / "meta.nc",
        run_root / "datasets" / "optics.nc",
        run_root / "datasets" / "phy.nc",
        run_root / "models" / "base" / "model.pkl",
        run_root / "models" / "base" / "selection.json",
        run_root / "models" / "final" / "model.pkl",
        run_root / "models" / "final" / "selection.json",
        run_root / "metrics" / "metrics.json",
        run_root / "metrics" / "predictions.csv",
        run_root / "reports" / "summary.md",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError("Synthetic run is missing expected artifacts: " + ", ".join(map(str, missing)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic inputs and run the full pipeline.")
    parser.add_argument("--config", default="configs/synthetic_end_to_end.yaml")
    parser.add_argument("--output-dir", default="examples/synthetic/generated")
    parser.add_argument("--run-id", help="Optional deterministic run id.")
    parser.add_argument("--n-observations", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_synthetic_example(args.output_dir, args.n_observations, args.seed)
    config, paths, state = initialize_run(args.config, run_id=args.run_id)
    run_pipeline(config, paths, state, overwrite=True, interactive=False)
    _assert_outputs(paths.root)
    print(f"Synthetic end-to-end run complete: {paths.root}")


if __name__ == "__main__":
    main()
