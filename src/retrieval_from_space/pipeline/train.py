from __future__ import annotations

from pathlib import Path

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.models.training import train_model
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def train(config: PipelineConfig, paths: RunPaths, state: PipelineState, interactive: bool = False) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.train", paths.logs / "train.log")
    state.mark("train", "running")
    artifacts = train_model(config, paths.root, interactive=interactive)
    for name, path in artifacts.items():
        logger.info("Saved %s: %s", name, path)
    state.mark("train", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
