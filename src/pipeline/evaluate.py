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


def _stage_summary(stage_metrics_path: Path) -> str:
    if not stage_metrics_path.exists():
        return ""
    payload = json.loads(stage_metrics_path.read_text(encoding="utf-8"))
    lines = ["\n## Stage Metrics\n", f"- Full stage report: `{stage_metrics_path}`"]
    for stage in payload.get("stages", []):
        metrics = stage.get("test_metrics") or stage.get("train_oof_metrics") or {}
        name = stage.get("name", "stage")
        role = stage.get("stage", "stage")
        lines.append(f"- {role} `{name}` test: {_compact_metric_line(metrics)}")
        if "train_oof_metrics" in stage:
            lines.append(
                f"- {role} `{name}` train OOF: {_compact_metric_line(stage['train_oof_metrics'])}"
            )
        elif "train_metrics" in stage:
            lines.append(f"- {role} `{name}` train: {_compact_metric_line(stage['train_metrics'])}")
    return "\n".join(lines) + "\n"


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
    stage_metrics_path = paths.metrics / "stage_metrics.json"
    report_path.write_text(
        "# Run Summary\n\n"
        f"- Problem type: {problem_type}\n"
        f"- Predictions: `{predictions_path}`\n"
        f"- Metrics: `{metrics_path}`\n\n"
        "```json\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "```\n"
        f"{_stage_summary(stage_metrics_path)}",
        encoding="utf-8",
    )
    logger.info("Saved metrics and report")
    artifacts = {"metrics": metrics_path, "summary": report_path}
    state.mark("evaluate", "complete", {k: str(v) for k, v in artifacts.items()})
    return artifacts
