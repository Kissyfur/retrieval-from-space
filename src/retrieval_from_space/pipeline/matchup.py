from __future__ import annotations

from pathlib import Path

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.data.matchups import create_product_matchups, save_matchups
from retrieval_from_space.data.targets import load_target_table, save_standard_target_table
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def create_matchups(config: PipelineConfig, paths: RunPaths, state: PipelineState, overwrite: bool = False) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.matchups", paths.logs / "matchups.log")
    state.mark("matchups", "running")
    targets = load_target_table(config.target)
    save_standard_target_table(targets, paths.processed / "targets.csv")

    artifacts: dict[str, Path] = {}
    for product in config.products:
        matchup_path = paths.matchups / f"{product.name}.nc"
        unmatched_path = paths.matchups / f"{product.name}_unmatched.csv"
        if matchup_path.exists() and not overwrite:
            logger.info("Skipping existing matchups %s", matchup_path)
            artifacts[product.name] = matchup_path
            continue
        raw_path = paths.raw / f"{product.name}.nc"
        local_raw_path = raw_path if product.source == "local" and raw_path.exists() else None
        logger.info("Creating matchups for %s", product.name)
        matchups, unmatched = create_product_matchups(
            product,
            targets,
            config.matchup,
            raw_path=local_raw_path,
        )
        save_matchups(matchups, unmatched, matchup_path, unmatched_path)
        if matchups is not None:
            artifacts[product.name] = matchup_path
            logger.info("Saved %s", matchup_path)
        logger.info("Unmatched observations for %s: %s", product.name, len(unmatched))
    state.mark("matchups", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
