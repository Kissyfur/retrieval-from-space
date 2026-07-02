from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.logging import setup_logger
from retrieval_from_space.metrics.classification import classification_metrics
from retrieval_from_space.metrics.regression import regression_metrics
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.state import PipelineState


def evaluate(config: PipelineConfig, paths: RunPaths, state: PipelineState) -> dict[str, Path]:
    logger = setup_logger("retrieval_from_space.evaluate", paths.logs / "evaluate.log")
    state.mark("evaluate", "running")
    predictions_path = paths.metrics / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    problem_type = config.problem.type or predictions["problem_type"].iloc[0]
    if problem_type == "classification":
        metrics = classification_metrics(predictions["y_true"], predictions["y_pred"])
    else:
        metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])

    metrics_path = paths.metrics / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = paths.reports / "summary.md"
    report_path.write_text(
        "# Run Summary\n\n"
        f"- Problem type: {problem_type}\n"
        f"- Predictions: `{predictions_path}`\n"
        f"- Metrics: `{metrics_path}`\n\n"
        "```json\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    logger.info("Saved metrics and report")
    artifacts = {"metrics": metrics_path, "summary": report_path}
    state.mark("evaluate", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
