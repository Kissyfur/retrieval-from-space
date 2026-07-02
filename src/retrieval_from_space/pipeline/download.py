from __future__ import annotations

from pathlib import Path

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.data.copernicus import open_copernicus_dataset, rename_common_dimensions, save_dataset
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def download_products(config: PipelineConfig, paths: RunPaths, state: PipelineState, overwrite: bool = False) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.download", paths.logs / "download.log")
    artifacts: dict[str, Path] = {}
    state.mark("download", "running")
    for product in config.products:
        output_path = paths.raw / f"{product.name}.nc"
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing raw product %s", output_path)
            artifacts[product.name] = output_path
            continue
        logger.info("Opening Copernicus product %s", product.name)
        ds = rename_common_dimensions(open_copernicus_dataset(product), product)
        artifacts[product.name] = save_dataset(ds, output_path)
        logger.info("Saved %s", output_path)
    state.mark("download", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
