from __future__ import annotations

from pathlib import Path

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.models.training import train_base_models, train_final_model, train_model
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def train(
    config: PipelineConfig,
    paths: RunPaths,
    state: PipelineState,
    interactive: bool = False,
    stage: str = "all",
    base_names: list[str] | None = None,
) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.train", paths.logs / "train.log")
    state_name = "train" if stage == "all" else f"train_{stage}"
    state.mark(state_name, "running")
    if stage == "all":
        artifacts = train_model(config, paths.root, interactive=interactive)
    elif stage == "base":
        artifacts = train_base_models(config, paths.root, base_names=base_names, interactive=interactive)
    elif stage == "final":
        artifacts = train_final_model(config, paths.root, interactive=interactive)
    else:
        raise ValueError("stage must be 'all', 'base', or 'final'.")
    for name, path in artifacts.items():
        logger.info("Saved %s: %s", name, path)
    state.mark(state_name, "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
