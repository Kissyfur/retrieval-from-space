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
from tqdm.auto import tqdm

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


def interval_soft_labeling(
    values: np.ndarray,
    intervals: list[list[float]],
    temperature: float = 1.0,
    prior: float = 1.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    intervals_arr = np.asarray(intervals, dtype=float)
    if intervals_arr.ndim != 2 or intervals_arr.shape[1] != 2:
        raise ValueError("Class intervals must be shaped as [[low, high], ...].")
    if temperature <= 0:
        raise ValueError("Soft-label temperature must be greater than zero.")
    if prior < 0:
        raise ValueError("Soft-label prior must be greater than or equal to zero.")

    left = intervals_arr[:, 0]
    right = intervals_arr[:, 1]
    distances = np.where(values < left, left - values, np.where(values > right, values - right, 0.0))
    similarities = np.exp(-distances / temperature) + prior
    return similarities / similarities.sum(axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    encoded = np.zeros((len(labels), n_classes), dtype=float)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


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


class ArrayStandardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> "ArrayStandardizer":
        axes = tuple(range(x.ndim - 1))
        self.mean_ = np.nanmean(x, axis=axes, keepdims=True)
        self.std_ = np.nanstd(x, axis=axes, keepdims=True)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The array standardizer has not been fitted.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def _stage_uses_cnn3d(stage: ModelStageConfig | ModelConfig) -> bool:
    return stage.family.lower() in {"cnn3d", "3d_cnn"}


def _stage_slug(name: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(name))
    return slug.strip("_") or "stage"


def _default_base_stage(config: ModelConfig) -> ModelStageConfig:
    return ModelStageConfig(
        family=config.family,
        feature_groups=config.feature_groups,
        params=config.params,
        standardize=config.standardize,
        sample_weight=config.sample_weight,
        augmentation=dict(config.augmentation),
        hyperparameter_search=config.hyperparameter_search,
    )


def _configured_base_stages(config: ModelConfig) -> dict[str, ModelStageConfig]:
    if config.base_models:
        return dict(config.base_models)
    if config.base_model is not None:
        return {"base": config.base_model}
    return {"base": _default_base_stage(config)}


def _load_group(path: Path, ids, flatten: bool = True) -> np.ndarray:
    data = xr.load_dataarray(path).sel(Id=ids)
    if not flatten:
        data = data.transpose("Id", "lat", "lon", "time", "variable")
        return np.asarray(data.values)
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
    for stage in _configured_base_stages(config).values():
        stage_paths.extend(_feature_group_paths(datasets_dir, stage.feature_groups))
    if config.final_model is not None:
        stage_paths.extend(_feature_group_paths(datasets_dir, config.final_model.feature_groups))
    if not stage_paths:
        return _feature_group_paths(datasets_dir, config.feature_groups)
    unique = {path.stem: path for path in stage_paths}
    return list(unique.values())


def _load_matrix_for_groups(
    datasets_dir: Path,
    ids,
    feature_groups: list[str],
    stage: ModelStageConfig | ModelConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    group_paths = _feature_group_paths(datasets_dir, feature_groups)
    missing = [path for path in group_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature group files: {missing}")
    if stage is not None and _stage_uses_cnn3d(stage):
        x = np.concatenate([_load_group(path, ids, flatten=False) for path in group_paths], axis=-1)
    else:
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
        y_reg = y.astype(float)
        return y_reg, y_reg, label_values, label_encoder

    if config.problem.class_intervals:
        hard_labels = interval_labeling(y.astype(float), config.problem.class_intervals)
        label_values = list(range(len(config.problem.class_intervals)))
        if config.problem.class_encoding == "one_hot":
            return _one_hot(hard_labels, len(label_values)), hard_labels, label_values, label_encoder
        if config.problem.class_encoding == "soft_probabilities":
            soft_labels = interval_soft_labeling(
                y.astype(float),
                config.problem.class_intervals,
                temperature=config.problem.soft_label_temperature,
                prior=config.problem.soft_label_prior,
            )
            return soft_labels, hard_labels, label_values, label_encoder
        return hard_labels, hard_labels, label_values, label_encoder

    label_encoder = LabelEncoder()
    hard_labels = label_encoder.fit_transform(y)
    label_values = list(range(len(label_encoder.classes_)))
    if config.problem.class_encoding == "one_hot":
        return _one_hot(hard_labels, len(label_values)), hard_labels, label_values, label_encoder
    if config.problem.class_encoding == "soft_probabilities":
        raise ValueError("Soft probability labels require problem.class_intervals.")
    return hard_labels, hard_labels, label_values, label_encoder


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


def _classification_labels(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        return np.argmax(prediction, axis=1)
    return prediction.reshape(-1)


def _validate_target_compatibility(problem_type: str, stage: ModelStageConfig, y: np.ndarray) -> None:
    if problem_type != "classification":
        return
    if np.asarray(y).ndim <= 1:
        return
    family = stage.family.lower()
    if family not in {"cnn", "cnn1d", "1d_cnn", "cnn3d", "3d_cnn"}:
        raise ValueError(
            "Probability-vector classification targets require a model family that accepts "
            "2D class targets, such as cnn3d. Use problem.class_encoding: hard for tree models."
        )


def _target_for_stage(
    problem_type: str,
    stage: ModelStageConfig,
    y_target: np.ndarray,
    y_labels: np.ndarray,
) -> np.ndarray:
    if problem_type == "classification" and np.asarray(y_target).ndim > 1 and not _stage_uses_cnn3d(stage):
        return y_labels
    return y_target


def _class_distribution(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    return {str(cls.item() if hasattr(cls, "item") else cls): int(count) for cls, count in zip(classes, counts)}


def _sample_weight_settings(stage: ModelStageConfig) -> dict[str, Any]:
    raw = stage.sample_weight
    if raw is None or raw is False:
        return {"enabled": False, "mode": "none", "class_boost": None}
    if raw is True:
        return {"enabled": True, "mode": "balanced", "class_boost": None}
    if isinstance(raw, str):
        return {"enabled": raw.lower() != "none", "mode": raw.lower(), "class_boost": None}
    if isinstance(raw, dict):
        mode = str(raw.get("mode", "balanced")).lower()
        return {
            "enabled": bool(raw.get("enabled", mode != "none")),
            "mode": mode,
            "class_boost": raw.get("class_boost", raw.get("class_boosts")),
        }
    raise ValueError("stage.sample_weight must be false, 'balanced', or a mapping.")


def _class_boost_for_label(class_boost, label) -> float:
    if class_boost is None:
        return 1.0
    if isinstance(class_boost, dict):
        return float(class_boost.get(str(label), class_boost.get(label, 1.0)))
    boosts = list(class_boost)
    index = int(label)
    return float(boosts[index]) if 0 <= index < len(boosts) else 1.0


def _make_sample_weights(
    problem_type: str,
    stage: ModelStageConfig,
    labels: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    settings = _sample_weight_settings(stage)
    if problem_type != "classification" or not settings["enabled"]:
        return None, settings
    if labels is None:
        raise ValueError("Classification sample weights require hard labels.")
    if settings["mode"] != "balanced":
        raise ValueError("Only sample_weight mode 'balanced' is currently supported.")

    labels = np.asarray(labels).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    base = {cls: len(labels) / (len(classes) * count) for cls, count in zip(classes, counts)}
    weights = np.array(
        [base[label] * _class_boost_for_label(settings["class_boost"], label) for label in labels],
        dtype=np.float32,
    )
    return weights, {
        **settings,
        "class_distribution": _class_distribution(labels),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
    }


def _noise_std_array(stage: ModelStageConfig, x: np.ndarray) -> tuple[np.ndarray | float, dict[str, Any]]:
    config = stage.augmentation
    std = config.get("noise_std", config.get("std", config.get("std_x", 0.0)))
    if isinstance(std, dict):
        values: list[Any] = []
        for group in stage.feature_groups:
            group_std = std.get(group, 0.0)
            if isinstance(group_std, list):
                values.extend(group_std)
            else:
                values.append(group_std)
        std = values
    info = {"requested": std, "adjustment": "none", "channels": int(x.shape[-1])}
    if isinstance(std, list):
        std_arr = np.asarray(std, dtype=np.float32)
        channels = x.shape[-1]
        if len(std_arr) == 1:
            return float(std_arr[0]), {**info, "requested_channels": 1}
        if len(std_arr) != channels:
            info = {**info, "requested_channels": int(len(std_arr))}
            if len(std_arr) > channels:
                std_arr = std_arr[:channels]
                info["adjustment"] = "truncated_to_feature_channels"
            else:
                std_arr = np.pad(std_arr, (0, channels - len(std_arr)), constant_values=0.0)
                info["adjustment"] = "padded_with_zero_noise"
        shape = (1,) * (x.ndim - 1) + (channels,)
        return std_arr.reshape(shape), info
    return float(std), info


def _augment_training_data(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None,
    sample_weight: np.ndarray | None,
    stage: ModelStageConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    config = stage.augmentation
    if not config or not bool(config.get("enabled", config.get("augment", False))):
        return x, y, labels, sample_weight, {"enabled": False, "input_samples": int(len(x)), "fit_samples": int(len(x))}

    repetitions = int(config.get("repetitions", 1))
    if repetitions <= 0:
        return x, y, labels, sample_weight, {"enabled": False, "input_samples": int(len(x)), "fit_samples": int(len(x))}

    rng = np.random.default_rng(int(config.get("seed", random_state)))
    x_aug = np.repeat(x, repetitions, axis=0).astype(np.float32, copy=True)
    noise_std, noise_info = _noise_std_array(stage, x_aug)
    if np.any(np.asarray(noise_std) != 0):
        x_aug = x_aug + rng.normal(0.0, noise_std, size=x_aug.shape)
    y_aug = np.repeat(y, repetitions, axis=0)
    labels_aug = None if labels is None else np.repeat(labels, repetitions, axis=0)
    weight_aug = None if sample_weight is None else np.repeat(sample_weight, repetitions, axis=0)

    x_fit = np.concatenate([x, x_aug], axis=0)
    y_fit = np.concatenate([y, y_aug], axis=0)
    labels_fit = None if labels is None else np.concatenate([labels, labels_aug], axis=0)
    weights_fit = None if sample_weight is None else np.concatenate([sample_weight, weight_aug], axis=0)
    return x_fit, y_fit, labels_fit, weights_fit, {
        "enabled": True,
        "input_samples": int(len(x)),
        "augmented_samples": int(len(x_aug)),
        "fit_samples": int(len(x_fit)),
        "repetitions": repetitions,
        "noise_std": config.get("noise_std", config.get("std", config.get("std_x", 0.0))),
        "noise_std_info": noise_info,
    }


def _fit_scaler_if_needed(
    x: np.ndarray,
    stage: ModelStageConfig,
) -> tuple[np.ndarray, StandardScaler | ArrayStandardizer | None]:
    if not stage.standardize:
        return x, None
    if _stage_uses_cnn3d(stage):
        scaler = ArrayStandardizer()
        return scaler.fit_transform(x), scaler
    scaler = StandardScaler()
    return scaler.fit_transform(x), scaler


def _transform_with_scaler(x: np.ndarray, scaler: StandardScaler | ArrayStandardizer | None) -> np.ndarray:
    return x if scaler is None else scaler.transform(x)


def _fit_estimator(
    problem_type: str,
    stage: ModelStageConfig,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    y_labels: np.ndarray | None = None,
    random_state: int = 42,
    progress_description: str | None = None,
):
    _validate_target_compatibility(problem_type, stage, y)
    x_proc, scaler = _fit_scaler_if_needed(x, stage)
    sample_weight, sample_weight_info = _make_sample_weights(problem_type, stage, y_labels)
    x_fit, y_fit, labels_fit, sample_weight_fit, augmentation_info = _augment_training_data(
        x_proc,
        y,
        y_labels,
        sample_weight,
        stage,
        random_state,
    )
    fit_params = dict(params)
    if progress_description and _stage_uses_cnn3d(stage):
        fit_params.setdefault("progress_description", progress_description)
    model = create_model(problem_type, stage, params=fit_params)
    if sample_weight_fit is None:
        model.fit(x_fit, y_fit)
    else:
        try:
            model.fit(x_fit, y_fit, sample_weight=sample_weight_fit)
        except TypeError:
            model.fit(x_fit, y_fit)
            sample_weight_info = {
                **sample_weight_info,
                "warning": f"{type(model).__name__}.fit did not accept sample_weight.",
            }
    model._retrieval_training_info = {
        "input_shape": list(x.shape),
        "fit_shape": list(x_fit.shape),
        "target_shape": list(np.asarray(y_fit).shape),
        "class_distribution": None if labels_fit is None else _class_distribution(labels_fit),
        "sample_weight": sample_weight_info,
        "augmentation": augmentation_info,
    }
    return model, scaler


def _predict_signal(problem_type: str, model, x: np.ndarray) -> np.ndarray:
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))
    prediction = np.asarray(model.predict(x))
    if problem_type == "classification" and prediction.ndim > 1:
        return prediction
    return prediction.reshape(-1, 1)


def _select_params(
    problem_type: str,
    stage: ModelStageConfig,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    split_labels: np.ndarray | None = None,
    score_labels: np.ndarray | None = None,
    stage_name: str = "model",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = stage.hyperparameter_search
    candidates = _candidate_pool(stage)
    if not search.enabled or len(candidates) == 1:
        return candidates[0], []

    scorer = get_scorer(search.scoring or _default_scoring(problem_type))
    splitter = _splitter(problem_type, search.cv, random_state)
    split_y = y if split_labels is None else split_labels
    score_y = y if score_labels is None else score_labels
    cv_results = []
    candidate_bar = tqdm(candidates, desc=f"{stage_name} hyperparameters", unit="candidate")
    for index, params in enumerate(candidate_bar):
        scores = []
        splits = list(splitter.split(x, split_y))
        fold_bar = tqdm(
            splits,
            desc=f"{stage_name} candidate {index + 1}/{len(candidates)} CV",
            unit="fold",
            leave=False,
        )
        for fold_index, (train_idx, val_idx) in enumerate(fold_bar, start=1):
            train_labels = split_y[train_idx] if problem_type == "classification" else None
            model, scaler = _fit_estimator(
                problem_type,
                stage,
                x[train_idx],
                y[train_idx],
                params,
                y_labels=train_labels,
                random_state=random_state + index,
                progress_description=(
                    f"{stage_name} candidate {index + 1}/{len(candidates)} "
                    f"fold {fold_index}/{len(splits)}"
                ),
            )
            x_val = _transform_with_scaler(x[val_idx], scaler)
            score = float(scorer(model, x_val, score_y[val_idx]))
            scores.append(score)
            fold_bar.set_postfix(score=f"{score:.4g}")
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_results.append(
            {
                "candidate_index": index,
                "params": params,
                "scores": scores,
                "mean_score": mean_score,
                "std_score": std_score,
            }
        )
        candidate_bar.set_postfix(best=f"{max(row['mean_score'] for row in cv_results):.4g}")
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
    split_labels: np.ndarray | None = None,
    stage_name: str = "model",
) -> np.ndarray:
    splitter = _splitter(problem_type, cv, random_state)
    split_y = y if split_labels is None else split_labels
    signal = None
    splits = list(splitter.split(x, split_y))
    for fold_index, (train_idx, val_idx) in enumerate(
        tqdm(splits, desc=f"{stage_name} OOF predictions", unit="fold"),
        start=1,
    ):
        train_labels = split_y[train_idx] if problem_type == "classification" else None
        model, scaler = _fit_estimator(
            problem_type,
            stage,
            x[train_idx],
            y[train_idx],
            params,
            y_labels=train_labels,
            random_state=random_state,
            progress_description=f"{stage_name} OOF fold {fold_index}/{len(splits)}",
        )
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


def _prediction_labels(problem_type: str, prediction: np.ndarray) -> np.ndarray:
    if problem_type == "classification":
        return _classification_labels(prediction)
    return np.asarray(prediction).reshape(-1)


def _metric_target(problem_type: str, y_target: np.ndarray, y_labels: np.ndarray) -> np.ndarray:
    return y_labels if problem_type == "classification" else y_target


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _save_model_artifact(model, path: Path) -> Path:
    if hasattr(model, "save"):
        return model.save(path)
    return save_pickle_model(model, path)


def _history_payload(model) -> dict[str, Any] | None:
    history = getattr(model, "history", None)
    if history is None:
        return None
    values = getattr(history, "history", None)
    if not values:
        return None
    payload = {key: [float(v) for v in series] for key, series in values.items()}
    payload["epochs_ran"] = len(next(iter(payload.values()))) if payload else 0
    return payload


def _history_summary(model) -> dict[str, Any] | None:
    payload = _history_payload(model)
    if not payload:
        return None
    summary = {"epochs_ran": payload.get("epochs_ran", 0)}
    for key, values in payload.items():
        if key == "epochs_ran" or not values:
            continue
        summary[f"final_{key}"] = values[-1]
        lower_is_better = any(token in key.lower() for token in ["loss", "error", "mse", "mae"])
        summary[f"best_{key}"] = min(values) if lower_is_better else max(values)
    return summary


def _training_info(model) -> dict[str, Any]:
    info = dict(getattr(model, "_retrieval_training_info", {}))
    history = _history_summary(model)
    if history:
        info["history"] = history
    return info


def _save_stage_artifacts(
    stage_dir: Path,
    model,
    scaler: StandardScaler | ArrayStandardizer | None,
    selected_params: dict[str, Any],
    cv_results: list[dict[str, Any]],
) -> dict[str, Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {"model": _save_model_artifact(model, stage_dir / "model.pkl")}
    if scaler is not None:
        with (stage_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        artifacts["scaler"] = stage_dir / "scaler.pkl"
    artifacts["selection"] = _write_json(
        stage_dir / "selection.json",
        {"selected_params": selected_params, "cv_results": cv_results},
    )
    history = _history_payload(model)
    if history:
        artifacts["history"] = _write_json(stage_dir / "history.json", history)
    return artifacts


def _save_training_report(paths: dict[str, Path], payload: dict[str, Any]) -> dict[str, Path]:
    return {
        "stage_metrics": _write_json(paths["metrics"] / "stage_metrics.json", payload),
        "training_report": _write_json(paths["reports"] / "training_report.json", payload),
    }


def _save_named_stage_metrics(
    paths: dict[str, Path],
    stage_name: str,
    payload: dict[str, Any],
) -> Path:
    return _write_json(paths["metrics"] / f"{_stage_slug(stage_name)}_metrics.json", payload)


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
    y_test_target,
    y_train_labels,
    y_test_labels,
    label_values,
    group_names: list[str],
) -> dict[str, Path]:
    stage = config.model
    selected_params, cv_results = _select_params(
        problem_type,
        stage,
        x_train,
        y_train,
        config.problem.random_state,
        split_labels=y_train_labels if problem_type == "classification" else None,
        score_labels=y_train_labels if problem_type == "classification" else None,
        stage_name="direct",
    )
    model, scaler = _fit_estimator(
        problem_type,
        stage,
        x_train,
        y_train,
        selected_params,
        y_labels=y_train_labels if problem_type == "classification" else None,
        random_state=config.problem.random_state,
        progress_description="direct final fit",
    )
    x_test_proc = _transform_with_scaler(x_test, scaler)
    raw_prediction = model.predict(x_test_proc)
    extra_prediction_columns = {}
    if problem_type == "classification":
        signal = _predict_signal(problem_type, model, x_test_proc)
        y_pred = _classification_labels(signal)
        if signal.ndim > 1 and signal.shape[1] > 1:
            extra_prediction_columns["class_probability"] = signal
        if np.asarray(y_test_target).ndim > 1:
            extra_prediction_columns["target_probability"] = y_test_target
    else:
        y_pred = raw_prediction
    metrics = _evaluate(problem_type, y_test_labels, y_pred, label_values)
    split_distribution = (
        {
            "train": _class_distribution(y_train_labels),
            "test": _class_distribution(y_test_labels),
        }
        if problem_type == "classification"
        else None
    )
    artifacts = _save_stage_artifacts(paths["models"] / "direct", model, scaler, selected_params, cv_results)
    artifacts["model"] = _save_model_artifact(model, paths["models"] / "model.pkl")
    direct_report = {
        "stage": "direct",
        "name": "direct",
        "family": stage.family,
        "feature_groups": group_names,
        "selected_params": selected_params,
        "cv_results": cv_results,
        "training": _training_info(model),
        "test_metrics": metrics,
    }
    artifacts["direct_metrics"] = _save_named_stage_metrics(paths, "direct", direct_report)
    artifacts.update(
        _save_training_report(
            paths,
            {
                "strategy": "direct",
                "problem_type": problem_type,
                "class_encoding": config.problem.class_encoding,
                "class_distribution": split_distribution,
                "stages": [direct_report],
                "final_metrics": metrics,
            },
        )
    )
    artifacts.update(
        _save_common_outputs(
            paths,
            problem_type,
            split_ids["test"],
            y_test_labels,
            y_pred,
            metrics,
            {
                "strategy": "direct",
                "train_ids": list(map(str, split_ids["train"])),
                "test_ids": list(map(str, split_ids["test"])),
                "feature_groups": group_names,
                "selected_params": selected_params,
                "class_distribution": split_distribution,
                "class_encoding": config.problem.class_encoding,
            },
            extra_prediction_columns=extra_prediction_columns,
        )
    )
    return artifacts


def _train_stacked_or_residual(
    config: PipelineConfig,
    problem_type: str,
    paths: dict[str, Path],
    split_ids,
    y_train,
    y_test_target,
    y_train_labels,
    y_test_labels,
    label_values,
) -> dict[str, Path]:
    strategy = config.model.strategy
    if strategy == "residual_correction" and problem_type != "regression":
        raise ValueError("Residual correction is only supported for regression problems.")
    base_stages = _configured_base_stages(config.model)
    if strategy == "residual_correction" and len(base_stages) != 1:
        raise ValueError("Residual correction supports exactly one base model.")
    final_stage = config.model.final_model or ModelStageConfig(feature_groups=["meta"])

    x_final_train_meta, final_groups = _load_matrix_for_groups(
        paths["datasets"], split_ids["train"], final_stage.feature_groups, final_stage
    )
    x_final_test_meta, _ = _load_matrix_for_groups(
        paths["datasets"], split_ids["test"], final_stage.feature_groups, final_stage
    )
    artifacts = {}
    base_train_signals: list[tuple[str, np.ndarray]] = []
    base_test_signals: list[tuple[str, np.ndarray]] = []
    base_reports: list[dict[str, Any]] = []
    base_group_payload: dict[str, list[str]] = {}
    base_param_payload: dict[str, dict[str, Any]] = {}

    train_metric_target = _metric_target(problem_type, y_train, y_train_labels)
    test_metric_target = _metric_target(problem_type, y_test_target, y_test_labels)

    base_stage_items = list(base_stages.items())
    for base_name, base_stage in tqdm(base_stage_items, desc="Base model stages", unit="stage"):
        slug = _stage_slug(base_name)
        stage_label = f"base:{base_name}"
        x_base_train, base_groups = _load_matrix_for_groups(
            paths["datasets"], split_ids["train"], base_stage.feature_groups, base_stage
        )
        x_base_test, _ = _load_matrix_for_groups(
            paths["datasets"], split_ids["test"], base_stage.feature_groups, base_stage
        )
        base_y_train = _target_for_stage(problem_type, base_stage, y_train, y_train_labels)

        base_params, base_cv_results = _select_params(
            problem_type,
            base_stage,
            x_base_train,
            base_y_train,
            config.problem.random_state,
            split_labels=y_train_labels if problem_type == "classification" else None,
            score_labels=y_train_labels if problem_type == "classification" else None,
            stage_name=stage_label,
        )
        oof_cv = max(2, base_stage.hyperparameter_search.cv if base_stage.hyperparameter_search.enabled else 5)
        base_oof_signal = _out_of_fold_signal(
            problem_type,
            base_stage,
            x_base_train,
            base_y_train,
            base_params,
            config.problem.random_state,
            oof_cv,
            split_labels=y_train_labels if problem_type == "classification" else None,
            stage_name=stage_label,
        )
        base_model, base_scaler = _fit_estimator(
            problem_type,
            base_stage,
            x_base_train,
            base_y_train,
            base_params,
            y_labels=y_train_labels if problem_type == "classification" else None,
            random_state=config.problem.random_state,
            progress_description=f"{stage_label} final fit",
        )
        base_test_signal = _predict_signal(
            problem_type, base_model, _transform_with_scaler(x_base_test, base_scaler)
        )

        base_train_pred = _prediction_labels(problem_type, base_oof_signal)
        base_test_pred = _prediction_labels(problem_type, base_test_signal)
        base_train_metrics = _evaluate(problem_type, train_metric_target, base_train_pred, label_values)
        base_test_metrics = _evaluate(problem_type, test_metric_target, base_test_pred, label_values)
        base_report = {
            "stage": "base",
            "name": base_name,
            "family": base_stage.family,
            "feature_groups": base_groups,
            "selected_params": base_params,
            "cv_results": base_cv_results,
            "oof_cv": oof_cv,
            "training": _training_info(base_model),
            "train_oof_metrics": base_train_metrics,
            "test_metrics": base_test_metrics,
        }
        base_reports.append(base_report)
        base_group_payload[base_name] = base_groups
        base_param_payload[base_name] = base_params
        base_train_signals.append((base_name, base_oof_signal))
        base_test_signals.append((base_name, base_test_signal))

        artifacts.update(
            {f"base_{slug}_{k}": v for k, v in _save_stage_artifacts(
                paths["models"] / "base" / slug,
                base_model,
                base_scaler,
                base_params,
                base_cv_results,
            ).items()}
        )
        artifacts[f"base_{slug}_metrics"] = _save_named_stage_metrics(
            paths, f"base_{slug}", base_report
        )

    if config.model.include_base_prediction:
        x_final_train = np.hstack([signal for _, signal in base_train_signals] + [x_final_train_meta])
        x_final_test = np.hstack([signal for _, signal in base_test_signals] + [x_final_test_meta])
    else:
        x_final_train = x_final_train_meta
        x_final_test = x_final_test_meta

    if strategy == "residual_correction":
        final_y_train = y_train - base_train_signals[0][1].reshape(-1)
        final_problem_type = "regression"
    else:
        final_problem_type = problem_type
        final_y_train = _target_for_stage(final_problem_type, final_stage, y_train, y_train_labels)

    final_params, final_cv_results = _select_params(
        final_problem_type,
        final_stage,
        x_final_train,
        final_y_train,
        config.problem.random_state,
        split_labels=y_train_labels if final_problem_type == "classification" else None,
        score_labels=y_train_labels if final_problem_type == "classification" else None,
        stage_name="final",
    )
    final_model, final_scaler = _fit_estimator(
        final_problem_type,
        final_stage,
        x_final_train,
        final_y_train,
        final_params,
        y_labels=y_train_labels if final_problem_type == "classification" else None,
        random_state=config.problem.random_state,
        progress_description="final fit",
    )
    x_final_test_proc = _transform_with_scaler(x_final_test, final_scaler)
    final_raw_pred = final_model.predict(x_final_test_proc)
    final_signal = None
    if strategy == "residual_correction":
        y_pred = base_test_signals[0][1].reshape(-1) + final_raw_pred
    elif problem_type == "classification":
        final_signal = _predict_signal(problem_type, final_model, x_final_test_proc)
        y_pred = _classification_labels(final_signal)
    else:
        y_pred = final_raw_pred

    x_final_train_proc = _transform_with_scaler(x_final_train, final_scaler)
    if strategy == "residual_correction":
        final_train_pred = base_train_signals[0][1].reshape(-1) + final_model.predict(x_final_train_proc)
    elif problem_type == "classification":
        final_train_pred = _classification_labels(_predict_signal(problem_type, final_model, x_final_train_proc))
    else:
        final_train_pred = final_model.predict(x_final_train_proc)

    metrics = _evaluate(problem_type, test_metric_target, y_pred, label_values)
    final_train_metrics = _evaluate(problem_type, train_metric_target, final_train_pred, label_values)
    split_distribution = (
        {
            "train": _class_distribution(y_train_labels),
            "test": _class_distribution(y_test_labels),
        }
        if problem_type == "classification"
        else None
    )
    artifacts.update(
        {f"final_{k}": v for k, v in _save_stage_artifacts(
            paths["models"] / "final", final_model, final_scaler, final_params, final_cv_results
        ).items()}
    )
    artifacts["model"] = artifacts["final_model"]
    base_prediction_columns = []
    extra_prediction_columns = {}
    for base_name, base_test_signal in base_test_signals:
        column_name = f"base_{_stage_slug(base_name)}_signal"
        extra_prediction_columns[column_name] = base_test_signal
        width = base_test_signal.shape[1] if base_test_signal.ndim > 1 else 1
        if width == 1:
            base_prediction_columns.append(column_name)
        else:
            base_prediction_columns.extend(f"{column_name}_{idx}" for idx in range(width))
    if problem_type == "classification" and strategy != "residual_correction":
        if final_signal is not None and final_signal.ndim > 1 and final_signal.shape[1] > 1:
            extra_prediction_columns["final_class_probability"] = final_signal
        if np.asarray(y_test_target).ndim > 1:
            extra_prediction_columns["target_probability"] = y_test_target
    final_report = {
        "stage": "final",
        "name": "final",
        "family": final_stage.family,
        "feature_groups": final_groups,
        "base_prediction_columns": base_prediction_columns,
        "selected_params": final_params,
        "cv_results": final_cv_results,
        "training": _training_info(final_model),
        "train_metrics": final_train_metrics,
        "test_metrics": metrics,
    }
    training_report = {
        "strategy": strategy,
        "problem_type": problem_type,
        "class_encoding": config.problem.class_encoding,
        "class_distribution": split_distribution,
        "base_models": {
            name: {
                "family": stage.family,
                "feature_groups": base_group_payload.get(name, stage.feature_groups),
            }
            for name, stage in base_stages.items()
        },
        "final_model": {
            "family": final_stage.family,
            "feature_groups": final_groups,
            "base_prediction_columns": base_prediction_columns,
        },
        "stages": base_reports + [final_report],
        "final_metrics": metrics,
    }
    artifacts["final_metrics"] = _save_named_stage_metrics(paths, "final", final_report)
    artifacts.update(_save_training_report(paths, training_report))
    artifacts.update(
        _save_common_outputs(
            paths,
            problem_type,
            split_ids["test"],
            test_metric_target,
            y_pred,
            metrics,
            {
                "strategy": strategy,
                "train_ids": list(map(str, split_ids["train"])),
                "test_ids": list(map(str, split_ids["test"])),
                "base_feature_groups": base_group_payload,
                "final_feature_groups": final_groups,
                "include_base_prediction": config.model.include_base_prediction,
                "base_selected_params": base_param_payload,
                "final_selected_params": final_params,
                "base_prediction_columns": base_prediction_columns,
                "class_distribution": split_distribution,
                "class_encoding": config.problem.class_encoding,
            },
            extra_prediction_columns=extra_prediction_columns,
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
    y_target, y_labels, label_values, label_encoder = _prepare_labels(problem_type, y_raw, config)
    stratify = y_labels if problem_type == "classification" else None
    train_ids, test_ids, y_train, y_test_target, y_train_labels, y_test_labels = train_test_split(
        ids,
        y_target,
        y_labels,
        test_size=config.problem.test_size,
        random_state=config.problem.random_state,
        stratify=stratify,
    )
    split_ids = {"train": train_ids, "test": test_ids}

    if label_encoder is not None:
        with (paths["models"] / "label_encoder.pkl").open("wb") as f:
            pickle.dump(label_encoder, f)

    if config.model.strategy == "direct":
        x_train, group_names = _load_matrix_for_groups(
            paths["datasets"], train_ids, config.model.feature_groups, config.model
        )
        x_test, _ = _load_matrix_for_groups(
            paths["datasets"], test_ids, config.model.feature_groups, config.model
        )
        return _train_direct(
            config,
            problem_type,
            paths,
            split_ids,
            x_train,
            x_test,
            y_train,
            y_test_target,
            y_train_labels,
            y_test_labels,
            label_values,
            group_names,
        )

    return _train_stacked_or_residual(
        config,
        problem_type,
        paths,
        split_ids,
        y_train,
        y_test_target,
        y_train_labels,
        y_test_labels,
        label_values,
    )
