from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.logging import setup_logger
from src.metrics.classification import (
    classification_metrics,
    labels_from_probabilities,
    save_confusion_matrix_plot,
    save_threshold_curve_plot,
    threshold_curve_metrics,
)
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
        y_pred = predictions["y_pred"]
        threshold_report = None
        threshold_config = dict(config.problem.decision_thresholds or {})
        probability_columns = sorted(
            [column for column in predictions.columns if column.startswith("class_probability_")],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if threshold_config.get("enabled") and probability_columns:
            class_index = int(threshold_config["class_index"])
            probabilities = predictions[probability_columns].to_numpy()
            thresholds = threshold_config.get("thresholds") or [
                round(value, 2) for value in [0.05 * index for index in range(1, 20)]
            ]
            rows = [
                {"split": "test", **row}
                for row in threshold_curve_metrics(
                    predictions["y_true"],
                    probabilities,
                    class_index,
                    thresholds,
                )
            ]
            threshold_csv = paths.metrics / f"decision_thresholds_class_{class_index}.csv"
            threshold_plot = paths.metrics / f"decision_thresholds_class_{class_index}.jpg"
            pd.DataFrame(rows).to_csv(threshold_csv, index=False)
            save_threshold_curve_plot(rows, threshold_plot, class_index)
            threshold = threshold_config.get("threshold")
            if threshold_config.get("apply", True) and threshold is not None:
                y_pred = labels_from_probabilities(probabilities, class_index, float(threshold))
            threshold_report = {
                "class_index": class_index,
                "threshold": threshold,
                "csv": str(threshold_csv),
                "plot": str(threshold_plot),
            }
        metrics = classification_metrics(predictions["y_true"], y_pred)
        save_confusion_matrix_plot(
            predictions["y_true"],
            y_pred,
            paths.metrics / "confusion_matrix.jpg",
            title="Confusion matrix",
        )
        save_confusion_matrix_plot(
            predictions["y_true"],
            y_pred,
            paths.metrics / "confusion_matrix_normalized_true.jpg",
            normalize="true",
            title="Confusion matrix normalized by true label",
        )
    else:
        metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
        threshold_report = None

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = paths.reports / "summary.md"
    threshold_line = (
        f"- Decision thresholds: `{threshold_report['csv']}`\n"
        if threshold_report
        else ""
    )
    report_path.write_text(
        "# Run Summary\n\n"
        f"- Problem type: {problem_type}\n"
        f"- Predictions: `{predictions_path}`\n"
        f"- Metrics: `{metrics_path}`\n"
        f"- Metric summary: {_compact_metric_line(metrics)}\n\n"
        f"{threshold_line}"
        "```json\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    logger.info("Saved metrics and report")
    artifacts = {"metrics": metrics_path, "summary": report_path}
    state.mark("evaluate", "complete", {key: str(value) for key, value in artifacts.items()})
    return artifacts
