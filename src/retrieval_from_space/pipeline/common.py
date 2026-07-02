from __future__ import annotations

from pathlib import Path

from retrieval_from_space import __version__
from retrieval_from_space.config import PipelineConfig, load_config, write_config_snapshot
from retrieval_from_space.paths import RunPaths, resolve_run_paths
from retrieval_from_space.state import PipelineState
import json
from datetime import datetime


def initialize_run(config_path: str | Path, run_id: str | None = None) -> tuple[PipelineConfig, RunPaths, PipelineState]:
    config = load_config(config_path)
    paths = resolve_run_paths(config.output_root, run_id or config.run_id, config.run_name, config.run_version)
    write_config_snapshot(config, paths.config / "config.json")
    write_run_manifest(config, paths)
    state = PipelineState.load(paths.state_file)
    return config, paths, state


def write_run_manifest(config: PipelineConfig, paths: RunPaths) -> Path:
    manifest_path = paths.root / "run_manifest.json"
    manifest = {
        "tool": "retrieval-from-space",
        "toolkit_version": __version__,
        "run_name": config.run_name,
        "run_version": config.run_version,
        "run_id": paths.root.name,
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(paths.root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
