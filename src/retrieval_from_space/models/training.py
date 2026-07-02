from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from retrieval_from_space.config import PipelineConfig
from retrieval_from_space.metrics.classification import classification_metrics
from retrieval_from_space.metrics.regression import regression_metrics
from retrieval_from_space.models.factory import create_model
from retrieval_from_space.models.tree import save_pickle_model


def interval_labeling(values: np.ndarray, intervals: list[list[float]]) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    intervals_arr = np.asarray(intervals, dtype=float)
    conditions = (values[:, None] >= intervals_arr[:, 0]) & (values[:, None] <= intervals_arr[:, 1])
    invalid = np.max(conditions, axis=1) == 0
    if np.any(invalid):
        raise ValueError(
            "Some target values do not belong to any configured class interval. "
            f"Range: {float(np.min(values))} to {float(np.max(values))}; intervals: {intervals}"
        )
    return np.argmax(conditions, axis=1)


def resolve_problem_type(config: PipelineConfig, interactive: bool = False) -> str:
    if config.problem.type:
        return config.problem.type
    if not interactive:
        raise ValueError("Set problem.type to 'classification' or 'regression' in the config.")
    answer = input("Is this a classification or regression problem? ").strip().lower()
    if answer not in {"classification", "regression"}:
        raise ValueError("Problem type must be 'classification' or 'regression'.")
    config.problem.type = answer
    return answer


def _load_group(path: Path, ids) -> np.ndarray:
    data = xr.load_dataarray(path).sel(Id=ids)
    return np.asarray(data.values).reshape(len(ids), -1)


def _ordered_intersection(reference_ids, group_paths: list[Path]) -> np.ndarray:
    common = set(reference_ids)
    for path in group_paths:
        ids = xr.load_dataarray(path).Id.values
        common &= set(ids)
    return np.array([value for value in reference_ids if value in common])


def _feature_group_paths(datasets_dir: Path, requested_groups: list[str]) -> list[Path]:
    if requested_groups:
        return [datasets_dir / f"{group}.nc" for group in requested_groups]
    return sorted(path for path in datasets_dir.glob("*.nc") if path.stem != "target")


def _load_training_matrix(datasets_dir: Path, config: PipelineConfig):
    target = xr.load_dataarray(datasets_dir / "target.nc")
    group_paths = _feature_group_paths(datasets_dir, config.model.feature_groups)
    missing = [path for path in group_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature group files: {missing}")

    ids = _ordered_intersection(target.Id.values, group_paths)
    if len(ids) == 0:
        raise ValueError("No common Id values found across target and feature groups.")

    y = np.asarray(target.sel(Id=ids).values).reshape(len(ids), -1)[:, 0]
    x = np.hstack([_load_group(path, ids) for path in group_paths])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, x, y, [path.stem for path in group_paths]


def _prepare_labels(problem_type: str, y: np.ndarray, config: PipelineConfig):
    label_encoder = None
    labels = None
    if problem_type == "regression":
        return y.astype(float), labels, label_encoder

    if config.problem.class_intervals:
        labels = interval_labeling(y.astype(float), config.problem.class_intervals)
        return labels, list(range(len(config.problem.class_intervals))), label_encoder

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    return labels, list(range(len(label_encoder.classes_))), label_encoder


def train_model(config: PipelineConfig, run_root: str | Path, interactive: bool = False) -> dict[str, Path]:
    problem_type = resolve_problem_type(config, interactive=interactive)
    run_root = Path(run_root)
    datasets_dir = run_root / "datasets"
    models_dir = run_root / "models"
    metrics_dir = run_root / "metrics"
    reports_dir = run_root / "reports"
    for path in (models_dir, metrics_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    ids, x, y_raw, group_names = _load_training_matrix(datasets_dir, config)
    y, label_values, label_encoder = _prepare_labels(problem_type, y_raw, config)
    stratify = y if problem_type == "classification" else None
    split = train_test_split(
        ids,
        x,
        y,
        test_size=config.problem.test_size,
        random_state=config.problem.random_state,
        stratify=stratify,
    )
    train_ids, test_ids, x_train, x_test, y_train, y_test = split

    scaler = None
    if config.model.standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    model = create_model(problem_type, config.model)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if problem_type == "classification":
        metrics = classification_metrics(y_test, y_pred, labels=label_values)
    else:
        metrics = regression_metrics(y_test, y_pred)

    model_path = save_pickle_model(model, models_dir / "model.pkl")
    if scaler is not None:
        with (models_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
    if label_encoder is not None:
        with (models_dir / "label_encoder.pkl").open("wb") as f:
            pickle.dump(label_encoder, f)

    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_path = metrics_dir / "predictions.csv"
    pd.DataFrame(
        {
            "Id": test_ids,
            "y_true": y_test,
            "y_pred": y_pred,
            "problem_type": problem_type,
        }
    ).to_csv(predictions_path, index=False)

    split_path = reports_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train_ids": list(map(str, train_ids)),
                "test_ids": list(map(str, test_ids)),
                "feature_groups": group_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model": model_path, "metrics": metrics_path, "predictions": predictions_path, "split": split_path}
