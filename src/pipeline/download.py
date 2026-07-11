from __future__ import annotations

import json
from pathlib import Path

from src.config import PipelineConfig, ProductSpec
from src.data.copernicus import open_copernicus_dataset, rename_common_dimensions, save_dataset
from src.logging import setup_logger
from src.paths import RunPaths
from src.state import PipelineState


def _write_remote_product_marker(product: ProductSpec, path: Path, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "name": product.name,
        "source": product.source,
        "dataset_ids": product.dataset_ids,
        "variables": product.variables,
        "open_dataset_kwargs": product.open_dataset_kwargs,
        "matchup": product.matchup,
        "preprocess": product.preprocess,
        "note": (
            "Remote Copernicus products are opened lazily during matchup creation. "
            "Only target-centered time/lat/lon windows are materialized as NetCDF matchups."
        ),
    }
    path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return path


def download_products(config: PipelineConfig, paths: RunPaths, state: PipelineState, overwrite: bool = False) -> dict[str, Path]:
    logger = setup_logger("src.download", paths.logs / "download.log")
    artifacts: dict[str, Path] = {}
    state.mark("download", "running")
    for product in config.products:
        if product.source != "local":
            marker_path = paths.raw / f"{product.name}.remote.json"
            logger.info(
                "Preparing remote product %s without materializing the full dataset; matchup stage will save only sliced windows.",
                product.name,
            )
            artifacts[product.name] = _write_remote_product_marker(product, marker_path, overwrite)
            continue

        output_path = paths.raw / f"{product.name}.nc"
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing raw product %s", output_path)
            artifacts[product.name] = output_path
            continue
        logger.info("Opening local product %s", product.name)
        ds = rename_common_dimensions(open_copernicus_dataset(product), product)
        artifacts[product.name] = save_dataset(ds, output_path)
        logger.info("Saved %s", output_path)
    state.mark("download", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
