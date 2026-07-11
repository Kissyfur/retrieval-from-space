from __future__ import annotations

from src.pipeline.download import download_products
from src.pipeline.evaluate import evaluate
from src.pipeline.matchup import create_matchups
from src.pipeline.preprocess import preprocess_datasets
from src.pipeline.train import train


def run_pipeline(config, paths, state, overwrite: bool = False, interactive: bool = False):
    download_products(config, paths, state, overwrite=overwrite)
    create_matchups(config, paths, state, overwrite=overwrite)
    preprocess_datasets(config, paths, state)
    train(config, paths, state, interactive=interactive)
    return evaluate(config, paths, state)
