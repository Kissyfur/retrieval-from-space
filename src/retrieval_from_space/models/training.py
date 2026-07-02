from __future__ import annotations

import json
import pickle
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from retrieval_from_space.config import ModelConfig, ModelStageConfig, PipelineConfig
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


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


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


def _all_feature_group_paths(datasets_dir: Path, config: ModelConfig) -> list[Path]:
    if config.strategy == "direct":
        return _feature_group_paths(datasets_dir, config.feature_groups)
    stage_paths = []
    for stage in (config.base_model, config.final_model):
        if stage is not None:
            stage_paths.extend(_feature_group_paths(datasets_dir, stage.feature_groups))
    if not stage_paths:
        return _feature_group_paths(datasets_dir, config.feature_groups)
    unique = {path.stem: path for path in stage_paths}
    return list(unique.values())


def _load_matrix_for_groups(datasets_dir: Path, ids, feature_groups: list[str]) -> tuple[np.ndarray, list[str]]:
    group_paths = _feature_group_paths(datasets_dir, feature_groups)
    missing = [path for path in group_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature group files: {missing}")
    x = np.hstack([_load_group(path, ids) for path in group_paths])
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), [path.stem for path in group_paths]


def _load_training_data(datasets_dir: Path, config: PipelineConfig):
    target = xr.load_dataarray(datasets_dir / "target.nc")
    group_paths = _all_feature_group_paths(datasets_dir, config.model)
    missing = [path for path in group_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature group files: {missing}")

    ids = _ordered_intersection(target.Id.values, group_paths)
    if len(ids) == 0:
        raise ValueError("No common Id values found across target and feature groups.")
    y_raw = np.asarray(target.sel(Id=ids).values).reshape(len(ids), -1)[:, 0]
    return ids, y_raw


def _prepare_labels(problem_type: str, y: np.ndarray, config: PipelineConfig):
    label_encoder = None
    label_values = None
    if problem_type == "regression":
        return y.astype(float), label_values, label_encoder

    if config.problem.class_intervals:
        labels = interval_labeling(y.astype(float), config.problem.class_intervals)
        return labels, list(range(len(config.problem.class_intervals))), label_encoder

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    return labels, list(range(len(label_encoder.classes_))), label_encoder


def _candidate_pool(stage: ModelStageConfig) -> list[dict[str, Any]]:
    search = stage.hyperparameter_search
    if not search.enabled:
        return [dict(stage.params)]
    candidates = [{**stage.params, **candidate} for candidate in search.candidates]
    if search.param_grid:
        keys = list(search.param_grid.keys())
        for values in product(*[search.param_grid[key] for key in keys]):
            candidates.append({**stage.params, **dict(zip(keys, values))})
    return candidates or [dict(stage.params)]


def _default_scoring(problem_type: str) -> str:
    return "f1_macro" if problem_type == "classification" else "r2"


def _splitter(problem_type: str, cv: int, random_state: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def _fit_scaler_if_needed(x: np.ndarray, stage: ModelStageConfig) -> tuple[np.ndarray, StandardScaler | None]:
    if not stage.standardize:
        return x, None
    scaler = StandardScaler()
    return scaler.fit_transform(x), scaler


def _transform_with_scaler(x: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    return x if scaler is None else scaler.transform(x)


def _fit_estimator(
    problem_type: str,
    stage: ModelStageConfig,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
):
    x_proc, scaler = _fit_scaler_if_needed(x, stage)
    model = create_model(problem_type, stage, params=params)
    model.fit(x_proc, y)
    return model, scaler


def _predict_signal(problem_type: str, model, x: np.ndarray) -> np.ndarray:
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))
    return np.asarray(model.predict(x)).reshape(-1, 1)


def _select_params(
    problem_type: str,
    stage: ModelStageConfig,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = stage.hyperparameter_search
    candidates = _candidate_pool(stage)
    if not search.enabled or len(candidates) == 1:
        return candidates[0], []

    scorer = get_scorer(search.scoring or _default_scoring(problem_type))
    splitter = _splitter(problem_type, search.cv, random_state)
    cv_results = []
    for index, params in enumerate(candidates):
        scores = []
        for train_idx, val_idx in splitter.split(x, y):
            model, scaler = _fit_estimator(problem_type, stage, x[train_idx], y[train_idx], params)
            x_val = _transform_with_scaler(x[val_idx], scaler)
            scores.append(float(scorer(model, x_val, y[val_idx])))
        cv_results.append(
            {
                "candidate_index": index,
                "params": params,
                "scores": scores,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
            }
        )
    best = max(cv_results, key=lambda row: row["mean_score"])
    return dict(best["params"]), cv_results


def _out_of_fold_signal(
    problem_type: str,
    stage: ModelStageConfig,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
    cv: int,
) -> np.ndarray:
    splitter = _splitter(problem_type, cv, random_state)
    signal = None
    for train_idx, val_idx in splitter.split(x, y):
        model, scaler = _fit_estimator(problem_type, stage, x[train_idx], y[train_idx], params)
        x_val = _transform_with_scaler(x[val_idx], scaler)
        fold_signal = _predict_signal(problem_type, model, x_val)
        if signal is None:
            signal = np.zeros((len(y), fold_signal.shape[1]), dtype=float)
        signal[val_idx] = fold_signal
    if signal is None:
        raise ValueError("Could not create out-of-fold predictions.")
    return signal


def _evaluate(problem_type: str, y_true, y_pred, label_values):
    if problem_type == "classification":
        return classification_metrics(y_true, y_pred, labels=label_values)
    return regression_metrics(y_true, y_pred)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _save_stage_artifacts(
    stage_dir: Path,
    model,
    scaler: StandardScaler | None,
    selected_params: dict[str, Any],
    cv_results: list[dict[str, Any]],
) -> dict[str, Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {"model": save_pickle_model(model, stage_dir / "model.pkl")}
    if scaler is not None:
        with (stage_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        artifacts["scaler"] = stage_dir / "scaler.pkl"
    artifacts["selection"] = _write_json(
        stage_dir / "selection.json",
        {"selected_params": selected_params, "cv_results": cv_results},
    )
    return artifacts


def _save_common_outputs(
    paths: dict[str, Path],
    problem_type: str,
    ids,
    y_test,
    y_pred,
    metrics: dict[str, Any],
    split_payload: dict[str, Any],
    extra_prediction_columns: dict[str, Any] | None = None,
) -> dict[str, Path]:
    metrics_path = paths["metrics"] / "metrics.json"
    predictions_path = paths["metrics"] / "predictions.csv"
    split_path = paths["reports"] / "split.json"

    _write_json(metrics_path, metrics)
    prediction_frame = pd.DataFrame(
        {
            "Id": ids,
            "y_true": y_test,
            "y_pred": y_pred,
            "problem_type": problem_type,
        }
    )
    for name, values in (extra_prediction_columns or {}).items():
        arr = np.asarray(values)
        if arr.ndim == 1 or arr.shape[1] == 1:
            prediction_frame[name] = arr.reshape(-1)
        else:
            for idx in range(arr.shape[1]):
                prediction_frame[f"{name}_{idx}"] = arr[:, idx]
    prediction_frame.to_csv(predictions_path, index=False)
    _write_json(split_path, split_payload)
    return {"metrics": metrics_path, "predictions": predictions_path, "split": split_path}


def _train_direct(
    config: PipelineConfig,
    problem_type: str,
    paths: dict[str, Path],
    split_ids,
    x_train,
    x_test,
    y_train,
    y_test,
    label_values,
    group_names: list[str],
) -> dict[str, Path]:
    stage = config.model
    selected_params, cv_results = _select_params(problem_type, stage, x_train, y_train, config.problem.random_state)
    model, scaler = _fit_estimator(problem_type, stage, x_train, y_train, selected_params)
    y_pred = model.predict(_transform_with_scaler(x_test, scaler))
    metrics = _evaluate(problem_type, y_test, y_pred, label_values)
    artifacts = _save_stage_artifacts(paths["models"] / "direct", model, scaler, selected_params, cv_results)
    artifacts["model"] = save_pickle_model(model, paths["models"] / "model.pkl")
    artifacts.update(
        _save_common_outputs(
            paths,
            problem_type,
            split_ids["test"],
            y_test,
            y_pred,
            metrics,
            {
                "strategy": "direct",
                "train_ids": list(map(str, split_ids["train"])),
                "test_ids": list(map(str, split_ids["test"])),
                "feature_groups": group_names,
                "selected_params": selected_params,
            },
        )
    )
    return artifacts


def _train_stacked_or_residual(
    config: PipelineConfig,
    problem_type: str,
    paths: dict[str, Path],
    split_ids,
    y_train,
    y_test,
    label_values,
) -> dict[str, Path]:
    strategy = config.model.strategy
    if strategy == "residual_correction" and problem_type != "regression":
        raise ValueError("Residual correction is only supported for regression problems.")
    base_stage = config.model.base_model or ModelStageConfig(
        family=config.model.family,
        feature_groups=config.model.feature_groups,
        params=config.model.params,
        standardize=config.model.standardize,
        hyperparameter_search=config.model.hyperparameter_search,
    )
    final_stage = config.model.final_model or ModelStageConfig(feature_groups=["meta"])

    x_base_train, base_groups = _load_matrix_for_groups(paths["datasets"], split_ids["train"], base_stage.feature_groups)
    x_base_test, _ = _load_matrix_for_groups(paths["datasets"], split_ids["test"], base_stage.feature_groups)
    x_final_train_meta, final_groups = _load_matrix_for_groups(paths["datasets"], split_ids["train"], final_stage.feature_groups)
    x_final_test_meta, _ = _load_matrix_for_groups(paths["datasets"], split_ids["test"], final_stage.feature_groups)

    base_params, base_cv_results = _select_params(
        problem_type, base_stage, x_base_train, y_train, config.problem.random_state
    )
    oof_cv = max(2, base_stage.hyperparameter_search.cv if base_stage.hyperparameter_search.enabled else 5)
    base_oof_signal = _out_of_fold_signal(
        problem_type, base_stage, x_base_train, y_train, base_params, config.problem.random_state, oof_cv
    )
    base_model, base_scaler = _fit_estimator(problem_type, base_stage, x_base_train, y_train, base_params)
    base_test_signal = _predict_signal(
        problem_type, base_model, _transform_with_scaler(x_base_test, base_scaler)
    )

    if config.model.include_base_prediction:
        x_final_train = np.hstack([base_oof_signal, x_final_train_meta])
        x_final_test = np.hstack([base_test_signal, x_final_test_meta])
    else:
        x_final_train = x_final_train_meta
        x_final_test = x_final_test_meta

    if strategy == "residual_correction":
        final_y_train = y_train - base_oof_signal.reshape(-1)
        final_problem_type = "regression"
    else:
        final_y_train = y_train
        final_problem_type = problem_type

    final_params, final_cv_results = _select_params(
        final_problem_type, final_stage, x_final_train, final_y_train, config.problem.random_state
    )
    final_model, final_scaler = _fit_estimator(final_problem_type, final_stage, x_final_train, final_y_train, final_params)
    final_raw_pred = final_model.predict(_transform_with_scaler(x_final_test, final_scaler))
    if strategy == "residual_correction":
        y_pred = base_test_signal.reshape(-1) + final_raw_pred
    else:
        y_pred = final_raw_pred

    metrics = _evaluate(problem_type, y_test, y_pred, label_values)
    artifacts = {}
    artifacts.update(
        {f"base_{k}": v for k, v in _save_stage_artifacts(
            paths["models"] / "base", base_model, base_scaler, base_params, base_cv_results
        ).items()}
    )
    artifacts.update(
        {f"final_{k}": v for k, v in _save_stage_artifacts(
            paths["models"] / "final", final_model, final_scaler, final_params, final_cv_results
        ).items()}
    )
    if strategy == "residual_correction":
        artifacts["model"] = artifacts["final_model"]
    artifacts.update(
        _save_common_outputs(
            paths,
            problem_type,
            split_ids["test"],
            y_test,
            y_pred,
            metrics,
            {
                "strategy": strategy,
                "train_ids": list(map(str, split_ids["train"])),
                "test_ids": list(map(str, split_ids["test"])),
                "base_feature_groups": base_groups,
                "final_feature_groups": final_groups,
                "include_base_prediction": config.model.include_base_prediction,
                "base_selected_params": base_params,
                "final_selected_params": final_params,
            },
            extra_prediction_columns={"base_signal": base_test_signal},
        )
    )
    return artifacts


def train_model(config: PipelineConfig, run_root: str | Path, interactive: bool = False) -> dict[str, Path]:
    problem_type = resolve_problem_type(config, interactive=interactive)
    run_root = Path(run_root)
    paths = {
        "datasets": run_root / "datasets",
        "models": run_root / "models",
        "metrics": run_root / "metrics",
        "reports": run_root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    ids, y_raw = _load_training_data(paths["datasets"], config)
    y, label_values, label_encoder = _prepare_labels(problem_type, y_raw, config)
    stratify = y if problem_type == "classification" else None
    train_ids, test_ids, y_train, y_test = train_test_split(
        ids,
        y,
        test_size=config.problem.test_size,
        random_state=config.problem.random_state,
        stratify=stratify,
    )
    split_ids = {"train": train_ids, "test": test_ids}

    if label_encoder is not None:
        with (paths["models"] / "label_encoder.pkl").open("wb") as f:
            pickle.dump(label_encoder, f)

    if config.model.strategy == "direct":
        x_train, group_names = _load_matrix_for_groups(paths["datasets"], train_ids, config.model.feature_groups)
        x_test, _ = _load_matrix_for_groups(paths["datasets"], test_ids, config.model.feature_groups)
        return _train_direct(
            config,
            problem_type,
            paths,
            split_ids,
            x_train,
            x_test,
            y_train,
            y_test,
            label_values,
            group_names,
        )

    return _train_stacked_or_residual(config, problem_type, paths, split_ids, y_train, y_test, label_values)
