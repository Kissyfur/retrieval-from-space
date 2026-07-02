from __future__ import annotations

from pathlib import Path

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.data.preprocessing import preprocess_matchups
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def preprocess_datasets(config: PipelineConfig, paths: RunPaths, state: PipelineState) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.preprocess", paths.logs / "preprocess.log")
    state.mark("preprocess", "running")
    artifacts = preprocess_matchups(config, paths.root)
    for name, path in artifacts.items():
        logger.info("Saved dataset %s: %s", name, path)
    state.mark("preprocess", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
