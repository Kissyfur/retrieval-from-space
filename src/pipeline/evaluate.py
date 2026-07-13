from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.logging import setup_logger
from src.metrics.classification import classification_metrics, save_confusion_matrix_plot
from src.metrics.regression import regression_metrics
from src.paths import RunPaths
from src.state import PipelineState


def _compact_metric_line(metrics: dict) -> str:
    keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "r2", "mae", "mse"]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, float)):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
    return ", ".join(parts) if parts else "metrics saved"


def evaluate(config: PipelineConfig, paths: RunPaths, state: PipelineState) -> dict[str, Path]:
    logger = setup_logger("src.evaluate", paths.logs / "evaluate.log")
    state.mark("evaluate", "running")
    predictions_path = paths.metrics / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    problem_type = config.problem.type or predictions["problem_type"].iloc[0]
    metrics_path = paths.metrics / "metrics.json"
    if problem_type == "classification":
        metrics = classification_metrics(predictions["y_true"], predictions["y_pred"])
        save_confusion_matrix_plot(
            predictions["y_true"],
            predictions["y_pred"],
            paths.metrics / "confusion_matrix.jpg",
            title="Confusion matrix",
        )
        save_confusion_matrix_plot(
            predictions["y_true"],
            predictions["y_pred"],
            paths.metrics / "confusion_matrix_normalized_true.jpg",
            normalize="true",
            title="Confusion matrix normalized by true label",
        )
    else:
        metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = paths.reports / "summary.md"
    report_path.write_text(
        "# Run Summary\n\n"
        f"- Problem type: {problem_type}\n"
        f"- Predictions: `{predictions_path}`\n"
        f"- Metrics: `{metrics_path}`\n"
        f"- Metric summary: {_compact_metric_line(metrics)}\n\n"
        "```json\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    logger.info("Saved metrics and report")
    artifacts = {"metrics": metrics_path, "summary": report_path}
    state.mark("evaluate", "complete", {key: str(value) for key, value in artifacts.items()})
    return artifacts
