from __future__ import annotations

from retrieval_from_space.pipeline.download import download_products
from retrieval_from_space.pipeline.evaluate import evaluate
from retrieval_from_space.pipeline.matchup import create_matchups
from retrieval_from_space.pipeline.preprocess import preprocess_datasets
from retrieval_from_space.pipeline.train import train


def run_pipeline(config, paths, state, overwrite: bool = False, interactive: bool = False):
    download_products(config, paths, state, overwrite=overwrite)
    create_matchups(config, paths, state, overwrite=overwrite)
    preprocess_datasets(config, paths, state)
    train(config, paths, state, interactive=interactive)
    return evaluate(config, paths, state)
