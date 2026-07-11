from __future__ import annotations

from pathlib import Path

from src.config import PipelineConfig
from src.data.preprocessing import preprocess_matchups
from src.logging import setup_logger
from src.paths import RunPaths
from src.state import PipelineState


def preprocess_datasets(config: PipelineConfig, paths: RunPaths, state: PipelineState) -> dict[str, Path]:
    logger = setup_logger("src.preprocess", paths.logs / "preprocess.log")
    state.mark("preprocess", "running")
    artifacts = preprocess_matchups(config, paths.root)
    for name, path in artifacts.items():
        logger.info("Saved dataset %s: %s", name, path)
    state.mark("preprocess", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
