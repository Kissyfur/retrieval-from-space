from __future__ import annotations

from pathlib import Path

from src.config import PipelineConfig
from src.logging import setup_logger
from src.models.training import train_model
from src.paths import RunPaths
from src.state import PipelineState


def train(
    config: PipelineConfig,
    paths: RunPaths,
    state: PipelineState,
    interactive: bool = False,
) -> dict[str, Path]:
    logger = setup_logger("src.train", paths.logs / "train.log")
    state.mark("train", "running")
    artifacts = train_model(config, paths.root, interactive=interactive)
    for name, path in artifacts.items():
        logger.info("Saved %s: %s", name, path)
    state.mark("train", "complete", {key: str(value) for key, value in artifacts.items()})
    return artifacts
