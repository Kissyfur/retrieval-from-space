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
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from retrieval_from_space.config import ModelConfig, ModelStageConfig, PipelineConfig
from retrieval_from_space.metrics.classification import classification_metrics, save_confusion_matrix_plot
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


def _effective_family(stage: ModelStageConfig | ModelConfig, params: dict[str, Any] | None = None) -> str:
    params = {} if params is None else params
    return str(params.get("family", params.get("model_family", stage.family))).lower()


def _stage_uses_cnn3d(
    stage: ModelStageConfig | ModelConfig,
    params: dict[str, Any] | None = None,
) -> bool:
    return _effective_family(stage, params) in {"cnn3d", "3d_cnn"}


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
        decision_thresholds=dict(config.decision_thresholds),
        input_selection=dict(config.input_selection),
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


def _splitter(problem_type: str, cv: int, random_state: int, repeats: int = 1):
    repeats = max(1, int(repeats))
    if problem_type == "classification":
        if repeats > 1:
            return RepeatedStratifiedKFold(
                n_splits=cv,
                n_repeats=repeats,
                random_state=random_state,
            )
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if repeats > 1:
        return RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def _classification_labels(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        return np.argmax(prediction, axis=1)
    return prediction.reshape(-1)


def _validate_target_compatibility(
    problem_type: str,
    stage: ModelStageConfig,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
) -> None:
    if problem_type != "classification":
        return
    if np.asarray(y).ndim <= 1:
        return
    family = _effective_family(stage, params)
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
    _validate_target_compatibility(problem_type, stage, y, params)
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
    splitter = _splitter(problem_type, search.cv, random_state, repeats=search.repeats)
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
                "cv": int(search.cv),
                "repeats": int(search.repeats),
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


def _threshold_settings(stage: ModelStageConfig) -> dict[str, Any]:
    raw = dict(stage.decision_thresholds or {})
    if not raw or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    return {
        "enabled": True,
        "tune": bool(raw.get("tune", raw.get("search", True))),
        "target_class": raw.get("target_class", raw.get("class")),
        "threshold": raw.get("threshold"),
        "grid": raw.get("grid", raw.get("thresholds")),
        "scoring": str(raw.get("scoring", "f1_macro")),
        "tie_tolerance": float(raw.get("tie_tolerance", raw.get("score_tolerance", 0.0))),
        "tie_breaker": str(raw.get("tie_breaker", "score")).lower(),
        "target_rate_penalty": float(raw.get("target_rate_penalty", 0.0)),
        "target_rate_multiplier": raw.get("target_rate_multiplier"),
        "target_rate_tolerance": raw.get("target_rate_tolerance"),
        "max_target_prediction_rate": raw.get("max_target_prediction_rate"),
    }


def _resolve_threshold_target_class(target_class, label_values, signal: np.ndarray) -> int:
    n_classes = int(np.asarray(signal).shape[1])
    if target_class is None:
        if label_values:
            return int(label_values[-1])
        return n_classes - 1
    target_index = int(target_class)
    if target_index < 0:
        target_index = n_classes + target_index
    if target_index < 0 or target_index >= n_classes:
        raise ValueError(
            f"decision_thresholds.target_class must be within 0..{n_classes - 1}; got {target_class}."
        )
    return target_index


def _threshold_grid(settings: dict[str, Any]) -> list[float]:
    if settings.get("threshold") is not None and not settings.get("tune", True):
        return [float(settings["threshold"])]
    raw_grid = settings.get("grid")
    if raw_grid is None:
        raw_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
    grid = [float(value) for value in raw_grid]
    if settings.get("threshold") is not None:
        grid.append(float(settings["threshold"]))
    unique = sorted({round(value, 10) for value in grid if 0.0 <= value <= 1.0})
    if not unique:
        raise ValueError("decision_thresholds.grid must contain at least one value between 0 and 1.")
    return unique


def _apply_target_class_threshold(
    signal: np.ndarray,
    target_class: int,
    threshold: float,
) -> np.ndarray:
    probabilities = np.asarray(signal)
    if probabilities.ndim != 2 or probabilities.shape[1] <= 1:
        return _classification_labels(probabilities)
    target_class = int(target_class)
    fallback = probabilities.copy()
    fallback[:, target_class] = -np.inf
    y_pred = np.argmax(fallback, axis=1)
    y_pred[probabilities[:, target_class] >= float(threshold)] = target_class
    return y_pred


def _classification_prediction_from_signal(
    signal: np.ndarray,
    decision_threshold_report: dict[str, Any] | None = None,
) -> np.ndarray:
    if decision_threshold_report and decision_threshold_report.get("enabled"):
        return _apply_target_class_threshold(
            signal,
            int(decision_threshold_report["target_class"]),
            float(decision_threshold_report["selected_threshold"]),
        )
    return _classification_labels(signal)


def _score_threshold_candidate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_values,
    scoring: str,
) -> tuple[float, dict[str, Any]]:
    metrics = classification_metrics(y_true, y_pred, labels=label_values)
    if scoring not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Unsupported decision threshold scoring '{scoring}'. Available: {available}.")
    return float(metrics[scoring]), metrics


def _class_metric_from_confusion_matrix(metrics: dict[str, Any], target_class: int) -> dict[str, float]:
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=float)
    if matrix.ndim != 2 or target_class < 0 or target_class >= matrix.shape[0]:
        return {
            "target_class_precision": 0.0,
            "target_class_recall": 0.0,
            "target_class_f1": 0.0,
        }
    true_positive = matrix[target_class, target_class]
    predicted = matrix[:, target_class].sum()
    actual = matrix[target_class, :].sum()
    precision = float(true_positive / predicted) if predicted else 0.0
    recall = float(true_positive / actual) if actual else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "target_class_precision": precision,
        "target_class_recall": recall,
        "target_class_f1": f1,
    }


def _select_threshold_candidate(candidates: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    score_key = "selection_score"
    best_score = max(row.get(score_key, row["score"]) for row in candidates)
    tolerance = max(float(settings.get("tie_tolerance", 0.0)), 0.0)
    contenders = [row for row in candidates if row.get(score_key, row["score"]) >= best_score - tolerance]
    tie_breaker = str(settings.get("tie_breaker", "score")).lower()
    if tie_breaker in {"target_recall", "target_class_recall", "recall"}:
        return max(
            contenders,
            key=lambda row: (
                row.get("target_class_recall", 0.0),
                row.get(score_key, row["score"]),
                row["score"],
                -row["threshold"],
            ),
        )
    if tie_breaker in {"target_f1", "target_class_f1"}:
        return max(
            contenders,
            key=lambda row: (
                row.get("target_class_f1", 0.0),
                row.get(score_key, row["score"]),
                row["score"],
                -row["threshold"],
            ),
        )
    if tie_breaker in {"lower_threshold", "lower"}:
        return min(contenders, key=lambda row: (row["threshold"], -row.get(score_key, row["score"])))
    if tie_breaker in {"higher_threshold", "higher"}:
        return max(contenders, key=lambda row: (row["threshold"], row.get(score_key, row["score"])))
    return max(contenders, key=lambda row: (row.get(score_key, row["score"]), row["score"], -row["threshold"]))


def _target_prediction_rate_limit(settings: dict[str, Any], target_actual_rate: float) -> float | None:
    if settings.get("max_target_prediction_rate") is not None:
        return float(settings["max_target_prediction_rate"])
    multiplier = settings.get("target_rate_multiplier")
    tolerance = settings.get("target_rate_tolerance")
    if multiplier is None and tolerance is None:
        return target_actual_rate if float(settings.get("target_rate_penalty", 0.0)) > 0 else None
    limit = target_actual_rate
    if multiplier is not None:
        limit = target_actual_rate * float(multiplier)
    if tolerance is not None:
        limit += float(tolerance)
    return max(0.0, min(1.0, float(limit)))


def _target_rate_penalty_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_class: int,
    settings: dict[str, Any],
) -> dict[str, float | None]:
    target_actual_rate = float(np.mean(np.asarray(y_true).reshape(-1) == target_class))
    target_prediction_rate = float(np.mean(np.asarray(y_pred).reshape(-1) == target_class))
    limit = _target_prediction_rate_limit(settings, target_actual_rate)
    penalty_weight = max(float(settings.get("target_rate_penalty", 0.0)), 0.0)
    excess = max(0.0, target_prediction_rate - limit) if limit is not None else 0.0
    penalty = penalty_weight * excess
    return {
        "target_actual_rate": target_actual_rate,
        "target_prediction_rate": target_prediction_rate,
        "target_prediction_rate_limit": limit,
        "target_prediction_rate_excess": excess,
        "target_rate_penalty_weight": penalty_weight,
        "target_rate_penalty": penalty,
    }


def _tune_decision_threshold(
    stage: ModelStageConfig,
    train_signal: np.ndarray,
    y_train_labels: np.ndarray,
    label_values,
) -> dict[str, Any] | None:
    settings = _threshold_settings(stage)
    if not settings["enabled"]:
        return None
    train_signal = np.asarray(train_signal)
    if train_signal.ndim != 2 or train_signal.shape[1] <= 1:
        return {
            "enabled": False,
            "reason": "Decision thresholds require class probability columns.",
        }

    target_class = _resolve_threshold_target_class(settings.get("target_class"), label_values, train_signal)
    scoring = settings["scoring"]
    candidates = []
    for threshold in _threshold_grid(settings):
        y_pred = _apply_target_class_threshold(train_signal, target_class, threshold)
        score, metrics = _score_threshold_candidate(y_train_labels, y_pred, label_values, scoring)
        target_metrics = _class_metric_from_confusion_matrix(metrics, target_class)
        rate_payload = _target_rate_penalty_payload(y_train_labels, y_pred, target_class, settings)
        selection_score = score - float(rate_payload["target_rate_penalty"])
        candidates.append(
            {
                "threshold": float(threshold),
                "score": score,
                "selection_score": selection_score,
                "metrics": metrics,
                **target_metrics,
                **rate_payload,
                "predicted_distribution": _class_distribution(y_pred),
            }
        )
    best_score = max(row["score"] for row in candidates)
    best_selection_score = max(row["selection_score"] for row in candidates)
    best = _select_threshold_candidate(candidates, settings)
    return {
        "enabled": True,
        "tuned": bool(settings.get("tune", True)),
        "target_class": int(target_class),
        "selected_threshold": float(best["threshold"]),
        "scoring": scoring,
        "selected_score": float(best["score"]),
        "selected_selection_score": float(best["selection_score"]),
        "primary_best_score": float(best_score),
        "selection_best_score": float(best_selection_score),
        "tie_tolerance": float(settings.get("tie_tolerance", 0.0)),
        "tie_breaker": settings.get("tie_breaker", "score"),
        "selected_target_class_recall": float(best.get("target_class_recall", 0.0)),
        "selected_target_class_f1": float(best.get("target_class_f1", 0.0)),
        "target_rate_penalty": {
            "weight": float(settings.get("target_rate_penalty", 0.0)),
            "multiplier": settings.get("target_rate_multiplier"),
            "tolerance": settings.get("target_rate_tolerance"),
            "max_prediction_rate": settings.get("max_target_prediction_rate"),
            "selected_target_prediction_rate": best.get("target_prediction_rate"),
            "selected_target_prediction_rate_limit": best.get("target_prediction_rate_limit"),
            "selected_penalty": best.get("target_rate_penalty"),
        },
        "candidates": candidates,
    }


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


def _ids_as_str(ids) -> np.ndarray:
    return np.asarray(list(map(str, ids)))


def _save_base_signals(
    paths: dict[str, Path],
    base_name: str,
    split_ids,
    train_signal: np.ndarray,
    test_signal: np.ndarray,
) -> Path:
    path = paths["metrics"] / f"base_{_stage_slug(base_name)}_signals.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_ids=_ids_as_str(split_ids["train"]),
        test_ids=_ids_as_str(split_ids["test"]),
        train_signal=np.asarray(train_signal),
        test_signal=np.asarray(test_signal),
    )
    return path


def _load_base_signals(paths: dict[str, Path], base_name: str, split_ids) -> tuple[np.ndarray, np.ndarray]:
    path = paths["metrics"] / f"base_{_stage_slug(base_name)}_signals.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing base signal file for {base_name}: {path}. "
            "Run `bin/train_model.py --stage base` first."
        )
    payload = np.load(path)
    expected_train = _ids_as_str(split_ids["train"])
    expected_test = _ids_as_str(split_ids["test"])
    if payload["train_ids"].tolist() != expected_train.tolist() or payload["test_ids"].tolist() != expected_test.tolist():
        raise ValueError(
            f"Base signal split for {base_name} does not match the current config/run split. "
            "Use the same config, run id, and random_state, or retrain the base stage."
        )
    return np.asarray(payload["train_signal"]), np.asarray(payload["test_signal"])


def _load_stage_report(paths: dict[str, Path], stage_name: str) -> dict[str, Any] | None:
    path = paths["metrics"] / f"{_stage_slug(stage_name)}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    artifacts = {"metrics": metrics_path, "predictions": predictions_path, "split": split_path}
    if problem_type == "classification":
        cm_path = paths["metrics"] / "confusion_matrix.jpg"
        cm_norm_path = paths["metrics"] / "confusion_matrix_normalized_true.jpg"
        save_confusion_matrix_plot(
            y_test,
            y_pred,
            cm_path,
            title="Confusion matrix",
        )
        save_confusion_matrix_plot(
            y_test,
            y_pred,
            cm_norm_path,
            normalize="true",
            title="Confusion matrix normalized by true label",
        )
        artifacts["confusion_matrix"] = cm_path
        artifacts["confusion_matrix_normalized_true"] = cm_norm_path
    return artifacts


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
        "family": _effective_family(stage, selected_params),
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


def _fit_base_stage(
    config: PipelineConfig,
    problem_type: str,
    paths: dict[str, Path],
    split_ids,
    y_train,
    y_test_target,
    y_train_labels,
    y_test_labels,
    label_values,
    base_name: str,
    base_stage: ModelStageConfig,
) -> dict[str, Any]:
    slug = _stage_slug(base_name)
    stage_label = f"base:{base_name}"
    train_metric_target = _metric_target(problem_type, y_train, y_train_labels)
    test_metric_target = _metric_target(problem_type, y_test_target, y_test_labels)

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
        "family": _effective_family(base_stage, base_params),
        "feature_groups": base_groups,
        "selected_params": base_params,
        "cv_results": base_cv_results,
        "oof_cv": oof_cv,
        "training": _training_info(base_model),
        "train_oof_metrics": base_train_metrics,
        "test_metrics": base_test_metrics,
    }

    artifacts = {
        f"base_{slug}_{key}": value
        for key, value in _save_stage_artifacts(
            paths["models"] / "base" / slug,
            base_model,
            base_scaler,
            base_params,
            base_cv_results,
        ).items()
    }
    artifacts[f"base_{slug}_metrics"] = _save_named_stage_metrics(paths, f"base_{slug}", base_report)
    artifacts[f"base_{slug}_signals"] = _save_base_signals(
        paths, base_name, split_ids, base_oof_signal, base_test_signal
    )
    return {
        "artifacts": artifacts,
        "report": base_report,
        "groups": base_groups,
        "params": base_params,
        "train_signal": base_oof_signal,
        "test_signal": base_test_signal,
    }


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
    artifacts = {}
    base_train_signals: list[tuple[str, np.ndarray]] = []
    base_test_signals: list[tuple[str, np.ndarray]] = []
    base_reports: list[dict[str, Any]] = []
    base_group_payload: dict[str, list[str]] = {}
    base_param_payload: dict[str, dict[str, Any]] = {}

    base_stage_items = list(base_stages.items())
    for base_name, base_stage in tqdm(base_stage_items, desc="Base model stages", unit="stage"):
        result = _fit_base_stage(
            config,
            problem_type,
            paths,
            split_ids,
            y_train,
            y_test_target,
            y_train_labels,
            y_test_labels,
            label_values,
            base_name,
            base_stage,
        )
        artifacts.update(result["artifacts"])
        base_report = result["report"]
        base_groups = result["groups"]
        base_params = result["params"]
        base_reports.append(base_report)
        base_group_payload[base_name] = base_groups
        base_param_payload[base_name] = base_params
        base_train_signals.append((base_name, result["train_signal"]))
        base_test_signals.append((base_name, result["test_signal"]))

    artifacts.update(
        _train_final_from_base_signals(
            config,
            problem_type,
            paths,
            split_ids,
            y_train,
            y_test_target,
            y_train_labels,
            y_test_labels,
            label_values,
            base_reports,
            base_train_signals,
            base_test_signals,
            base_group_payload,
            base_param_payload,
        )
    )
    return artifacts


def _training_paths(run_root: str | Path) -> dict[str, Path]:
    run_root = Path(run_root)
    paths = {
        "datasets": run_root / "datasets",
        "models": run_root / "models",
        "metrics": run_root / "metrics",
        "reports": run_root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _prepare_training_inputs(config: PipelineConfig, paths: dict[str, Path], interactive: bool = False):
    problem_type = resolve_problem_type(config, interactive=interactive)
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
    return {
        "problem_type": problem_type,
        "split_ids": split_ids,
        "y_train": y_train,
        "y_test_target": y_test_target,
        "y_train_labels": y_train_labels,
        "y_test_labels": y_test_labels,
        "label_values": label_values,
    }


def train_base_models(
    config: PipelineConfig,
    run_root: str | Path,
    base_names: list[str] | None = None,
    interactive: bool = False,
) -> dict[str, Path]:
    paths = _training_paths(run_root)
    context = _prepare_training_inputs(config, paths, interactive=interactive)
    problem_type = context["problem_type"]
    if config.model.strategy == "direct":
        raise ValueError("Base-stage training is only available for stacked or residual strategies.")
    base_stages = _configured_base_stages(config.model)
    selected_names = list(base_stages) if not base_names else base_names
    missing = [name for name in selected_names if name not in base_stages]
    if missing:
        raise ValueError(f"Unknown base model name(s): {missing}. Available: {list(base_stages)}")

    artifacts: dict[str, Path] = {}
    reports = []
    group_payload = {}
    param_payload = {}
    for base_name in tqdm(selected_names, desc="Requested base stages", unit="stage"):
        result = _fit_base_stage(
            config,
            problem_type,
            paths,
            context["split_ids"],
            context["y_train"],
            context["y_test_target"],
            context["y_train_labels"],
            context["y_test_labels"],
            context["label_values"],
            base_name,
            base_stages[base_name],
        )
        artifacts.update(result["artifacts"])
        reports.append(result["report"])
        group_payload[base_name] = result["groups"]
        param_payload[base_name] = result["params"]

    stage_report = {
        "strategy": config.model.strategy,
        "stage": "base",
        "problem_type": problem_type,
        "class_encoding": config.problem.class_encoding,
        "trained_base_models": selected_names,
        "base_feature_groups": group_payload,
        "base_selected_params": param_payload,
        "stages": reports,
    }
    artifacts.update(
        {
            "base_stage_report": _write_json(paths["reports"] / "base_training_report.json", stage_report),
            "base_stage_metrics": _write_json(paths["metrics"] / "base_stage_metrics.json", stage_report),
        }
    )
    return artifacts


def _load_base_reports_and_signals(config: PipelineConfig, paths: dict[str, Path], split_ids):
    base_stages = _configured_base_stages(config.model)
    base_reports = []
    base_train_signals = []
    base_test_signals = []
    base_group_payload = {}
    base_param_payload = {}
    for base_name, base_stage in base_stages.items():
        train_signal, test_signal = _load_base_signals(paths, base_name, split_ids)
        base_train_signals.append((base_name, train_signal))
        base_test_signals.append((base_name, test_signal))
        report = _load_stage_report(paths, f"base_{_stage_slug(base_name)}") or {
            "stage": "base",
            "name": base_name,
            "family": base_stage.family,
            "feature_groups": base_stage.feature_groups,
        }
        base_reports.append(report)
        base_group_payload[base_name] = report.get("feature_groups", base_stage.feature_groups)
        base_param_payload[base_name] = report.get("selected_params", {})
    return base_reports, base_train_signals, base_test_signals, base_group_payload, base_param_payload


def _matrix_column_names(prefix: str, values: np.ndarray) -> list[str]:
    values = np.asarray(values)
    width = values.shape[1] if values.ndim > 1 else 1
    if width == 1:
        return [prefix]
    return [f"{prefix}_{idx}" for idx in range(width)]


def _base_signal_columns(base_name: str, signal: np.ndarray) -> list[str]:
    return _matrix_column_names(f"base_{_stage_slug(base_name)}_signal", signal)


def _metadata_feature_columns(datasets_dir: Path, group_names: list[str], fallback_width: int) -> list[str]:
    columns: list[str] = []
    for group_name in group_names:
        path = datasets_dir / f"{group_name}.nc"
        if not path.exists():
            continue
        data = xr.load_dataarray(path)
        if "variable" not in data.coords:
            continue
        names = [str(value) for value in data["variable"].values.tolist()]
        if data.ndim == 2:
            columns.extend(names)
        else:
            columns.extend(f"{group_name}:{name}" for name in names)
    if len(columns) == fallback_width:
        return columns
    return [f"meta_{idx}" for idx in range(fallback_width)]


def _input_selection_settings(stage: ModelStageConfig, problem_type: str) -> dict[str, Any]:
    raw = dict(stage.input_selection or {})
    if not raw or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    return {
        "enabled": True,
        "scoring": str(raw.get("scoring", _default_scoring(problem_type))),
        "cv": raw.get("cv"),
        "candidates": raw.get("candidates", raw.get("variants")),
        "tie_tolerance": float(raw.get("tie_tolerance", raw.get("score_tolerance", 0.0))),
        "tie_breaker": str(raw.get("tie_breaker", "score")).lower(),
        "preferred_variants": [str(name) for name in raw.get("preferred_variants", [])],
    }


def _final_feature_variants(
    x_meta_train: np.ndarray,
    x_meta_test: np.ndarray,
    meta_columns: list[str],
    base_train_signals: list[tuple[str, np.ndarray]],
    base_test_signals: list[tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "metadata_only",
            "x_train": x_meta_train,
            "x_test": x_meta_test,
            "feature_columns": meta_columns,
        }
    ]
    if not base_train_signals:
        return variants

    base_train_all = np.hstack([signal for _, signal in base_train_signals])
    base_test_all = np.hstack([signal for _, signal in base_test_signals])
    base_columns = [
        column
        for base_name, signal in base_train_signals
        for column in _base_signal_columns(base_name, signal)
    ]
    variants.append(
        {
            "name": "base_signals_only",
            "x_train": base_train_all,
            "x_test": base_test_all,
            "feature_columns": base_columns,
        }
    )
    variants.append(
        {
            "name": "all_signals_plus_metadata",
            "x_train": np.hstack([base_train_all, x_meta_train]),
            "x_test": np.hstack([base_test_all, x_meta_test]),
            "feature_columns": base_columns + meta_columns,
        }
    )
    for (base_name, train_signal), (_, test_signal) in zip(base_train_signals, base_test_signals):
        signal_columns = _base_signal_columns(base_name, train_signal)
        slug = _stage_slug(base_name)
        variants.append(
            {
                "name": f"{slug}_signal_only",
                "x_train": train_signal,
                "x_test": test_signal,
                "feature_columns": signal_columns,
            }
        )
        variants.append(
            {
                "name": f"{slug}_signal_plus_metadata",
                "x_train": np.hstack([train_signal, x_meta_train]),
                "x_test": np.hstack([test_signal, x_meta_test]),
                "feature_columns": signal_columns + meta_columns,
            }
        )
    return variants


def _default_final_variant_name(include_base_prediction: bool, variants: list[dict[str, Any]]) -> str:
    names = {variant["name"] for variant in variants}
    if include_base_prediction and "all_signals_plus_metadata" in names:
        return "all_signals_plus_metadata"
    return "metadata_only"


def _filter_final_input_variants(
    variants: list[dict[str, Any]],
    requested: Any,
) -> list[dict[str, Any]]:
    if not requested:
        return variants
    requested_names = [str(name) for name in requested]
    by_name = {variant["name"]: variant for variant in variants}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown final input selection candidate(s): {missing}. Available: {list(by_name)}")
    return [by_name[name] for name in requested_names]


def _score_from_metrics(metrics: dict[str, Any], scoring: str) -> float:
    if scoring not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Unsupported scoring '{scoring}'. Available: {available}.")
    return float(metrics[scoring])


def _input_selection_sort_key(row: dict[str, Any], settings: dict[str, Any]):
    tie_breaker = str(settings.get("tie_breaker", "score")).lower()
    if tie_breaker in {"target_recall", "target_class_recall", "recall"}:
        return (row.get("target_class_recall", 0.0), row["score"], -row["feature_count"])
    if tie_breaker in {"target_f1", "target_class_f1"}:
        return (row.get("target_class_f1", 0.0), row["score"], -row["feature_count"])
    if tie_breaker in {"simpler", "fewest_features", "feature_count"}:
        return (-row["feature_count"], row["score"])
    return (row["score"], -row["feature_count"])


def _select_input_selection_row(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    best_score = max(row["score"] for row in rows)
    tolerance = max(float(settings.get("tie_tolerance", 0.0)), 0.0)
    contenders = [row for row in rows if row["score"] >= best_score - tolerance]
    tie_breaker = str(settings.get("tie_breaker", "score")).lower()
    if tie_breaker in {"preferred", "preferred_variant", "preferred_variants", "priority"}:
        by_name = {row["name"]: row for row in contenders}
        for name in settings.get("preferred_variants", []):
            if name in by_name:
                return by_name[name]
    return max(contenders, key=lambda row: _input_selection_sort_key(row, settings))


def _evaluate_final_variant_oof(
    config: PipelineConfig,
    problem_type: str,
    final_problem_type: str,
    final_stage: ModelStageConfig,
    final_y_train: np.ndarray,
    y_train_labels: np.ndarray,
    train_metric_target: np.ndarray,
    label_values,
    variant: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    selected_params, cv_results = _select_params(
        final_problem_type,
        final_stage,
        variant["x_train"],
        final_y_train,
        config.problem.random_state,
        split_labels=y_train_labels if final_problem_type == "classification" else None,
        score_labels=y_train_labels if final_problem_type == "classification" else None,
        stage_name=f"final:{variant['name']}",
    )
    configured_cv = settings.get("cv")
    oof_cv = int(configured_cv) if configured_cv else max(
        2,
        final_stage.hyperparameter_search.cv if final_stage.hyperparameter_search.enabled else 5,
    )
    oof_signal = _out_of_fold_signal(
        final_problem_type,
        final_stage,
        variant["x_train"],
        final_y_train,
        selected_params,
        config.problem.random_state,
        oof_cv,
        split_labels=y_train_labels if final_problem_type == "classification" else None,
        stage_name=f"final:{variant['name']}",
    )
    decision_threshold_report = None
    if final_problem_type == "classification":
        decision_threshold_report = _tune_decision_threshold(
            final_stage,
            oof_signal,
            y_train_labels,
            label_values,
        )
        oof_pred = _classification_prediction_from_signal(oof_signal, decision_threshold_report)
    else:
        oof_pred = _prediction_labels(final_problem_type, oof_signal)
    metrics = _evaluate(problem_type, train_metric_target, oof_pred, label_values)
    score = _score_from_metrics(metrics, settings["scoring"])
    target_metrics = {}
    if final_problem_type == "classification":
        target_class = (
            decision_threshold_report["target_class"]
            if decision_threshold_report and decision_threshold_report.get("enabled")
            else (label_values[-1] if label_values else np.asarray(oof_signal).shape[1] - 1)
        )
        target_metrics = _class_metric_from_confusion_matrix(metrics, int(target_class))
    return {
        "name": variant["name"],
        "feature_count": int(variant["x_train"].shape[1]),
        "feature_columns": variant["feature_columns"],
        "selected_params": selected_params,
        "cv_results": cv_results,
        "oof_cv": oof_cv,
        "oof_metrics": metrics,
        "score": score,
        "decision_thresholds": decision_threshold_report,
        **target_metrics,
    }


def _select_final_input_variant(
    config: PipelineConfig,
    problem_type: str,
    final_problem_type: str,
    final_stage: ModelStageConfig,
    variants: list[dict[str, Any]],
    final_y_train: np.ndarray,
    y_train_labels: np.ndarray,
    train_metric_target: np.ndarray,
    label_values,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    settings = _input_selection_settings(final_stage, problem_type)
    by_name = {variant["name"]: variant for variant in variants}
    if not settings["enabled"] or config.model.strategy != "stacking":
        selected_name = _default_final_variant_name(config.model.include_base_prediction, variants)
        return by_name[selected_name], None, None

    candidate_variants = _filter_final_input_variants(variants, settings.get("candidates"))
    rows = []
    for variant in tqdm(candidate_variants, desc="Final input variants", unit="variant"):
        rows.append(
            _evaluate_final_variant_oof(
                config,
                problem_type,
                final_problem_type,
                final_stage,
                final_y_train,
                y_train_labels,
                train_metric_target,
                label_values,
                variant,
                settings,
            )
        )
    selected = _select_input_selection_row(rows, settings)
    report = {
        "enabled": True,
        "scoring": settings["scoring"],
        "tie_tolerance": settings["tie_tolerance"],
        "tie_breaker": settings["tie_breaker"],
        "preferred_variants": settings.get("preferred_variants", []),
        "hyperparameter_search": {
            "enabled": bool(final_stage.hyperparameter_search.enabled),
            "cv": int(final_stage.hyperparameter_search.cv),
            "repeats": int(final_stage.hyperparameter_search.repeats),
            "candidate_count": len(_candidate_pool(final_stage)),
        },
        "selected_variant": selected["name"],
        "selected_score": float(selected["score"]),
        "selected_params": selected["selected_params"],
        "decision_thresholds": selected.get("decision_thresholds"),
        "candidates": rows,
    }
    return by_name[selected["name"]], report, selected


def _final_variant_metrics(
    config: PipelineConfig,
    problem_type: str,
    final_problem_type: str,
    final_stage: ModelStageConfig,
    final_y_train: np.ndarray,
    y_train_labels: np.ndarray,
    train_metric_target: np.ndarray,
    test_metric_target: np.ndarray,
    label_values,
    final_params: dict[str, Any],
    decision_threshold_report: dict[str, Any] | None,
    variant: dict[str, Any],
) -> dict[str, Any]:
    model, scaler = _fit_estimator(
        final_problem_type,
        final_stage,
        variant["x_train"],
        final_y_train,
        final_params,
        y_labels=y_train_labels if final_problem_type == "classification" else None,
        random_state=config.problem.random_state,
        progress_description=f"final ablation {variant['name']}",
    )
    x_train_proc = _transform_with_scaler(variant["x_train"], scaler)
    x_test_proc = _transform_with_scaler(variant["x_test"], scaler)
    if final_problem_type == "classification":
        train_signal = _predict_signal(problem_type, model, x_train_proc)
        test_signal = _predict_signal(problem_type, model, x_test_proc)
        train_pred = _classification_prediction_from_signal(train_signal, decision_threshold_report)
        test_pred = _classification_prediction_from_signal(test_signal, decision_threshold_report)
        argmax_test_pred = _classification_labels(test_signal)
        payload = {
            "name": variant["name"],
            "feature_count": int(variant["x_train"].shape[1]),
            "feature_columns": variant["feature_columns"],
            "train_metrics": _evaluate(problem_type, train_metric_target, train_pred, label_values),
            "test_metrics": _evaluate(problem_type, test_metric_target, test_pred, label_values),
            "test_metrics_argmax": _evaluate(problem_type, test_metric_target, argmax_test_pred, label_values),
        }
    else:
        train_pred = model.predict(x_train_proc)
        test_pred = model.predict(x_test_proc)
        payload = {
            "name": variant["name"],
            "feature_count": int(variant["x_train"].shape[1]),
            "feature_columns": variant["feature_columns"],
            "train_metrics": _evaluate(problem_type, train_metric_target, train_pred, label_values),
            "test_metrics": _evaluate(problem_type, test_metric_target, test_pred, label_values),
        }
    training = _training_info(model)
    if training:
        payload["training"] = training
    return payload


def _save_final_ablation_report(
    config: PipelineConfig,
    problem_type: str,
    final_problem_type: str,
    final_stage: ModelStageConfig,
    paths: dict[str, Path],
    x_final_train_meta: np.ndarray,
    x_final_test_meta: np.ndarray,
    final_groups: list[str],
    base_train_signals: list[tuple[str, np.ndarray]],
    base_test_signals: list[tuple[str, np.ndarray]],
    final_y_train: np.ndarray,
    y_train_labels: np.ndarray,
    train_metric_target: np.ndarray,
    test_metric_target: np.ndarray,
    label_values,
    final_params: dict[str, Any],
    decision_threshold_report: dict[str, Any] | None,
) -> Path | None:
    if config.model.strategy != "stacking":
        return None
    meta_columns = _metadata_feature_columns(paths["datasets"], final_groups, x_final_train_meta.shape[1])
    variants = _final_feature_variants(
        x_final_train_meta,
        x_final_test_meta,
        meta_columns,
        base_train_signals,
        base_test_signals,
    )
    reports = []
    for variant in tqdm(variants, desc="Final ablation variants", unit="variant"):
        reports.append(
            _final_variant_metrics(
                config,
                problem_type,
                final_problem_type,
                final_stage,
                final_y_train,
                y_train_labels,
                train_metric_target,
                test_metric_target,
                label_values,
                final_params,
                decision_threshold_report,
                variant,
            )
        )
    return _write_json(
        paths["metrics"] / "final_ablation_metrics.json",
        {
            "stage": "final_ablation",
            "strategy": config.model.strategy,
            "problem_type": problem_type,
            "selected_params": final_params,
            "decision_thresholds_applied": decision_threshold_report,
            "notes": "Ablation train metrics are in-sample diagnostics; use test metrics for comparison.",
            "variants": reports,
        },
    )


def _train_final_from_base_signals(
    config: PipelineConfig,
    problem_type: str,
    paths: dict[str, Path],
    split_ids,
    y_train,
    y_test_target,
    y_train_labels,
    y_test_labels,
    label_values,
    base_reports,
    base_train_signals,
    base_test_signals,
    base_group_payload,
    base_param_payload,
) -> dict[str, Path]:
    strategy = config.model.strategy
    if strategy == "residual_correction" and problem_type != "regression":
        raise ValueError("Residual correction is only supported for regression problems.")
    if strategy == "residual_correction" and len(base_train_signals) != 1:
        raise ValueError("Residual correction supports exactly one base model.")
    final_stage = config.model.final_model or ModelStageConfig(feature_groups=["meta"])
    x_final_train_meta, final_groups = _load_matrix_for_groups(
        paths["datasets"], split_ids["train"], final_stage.feature_groups, final_stage
    )
    x_final_test_meta, _ = _load_matrix_for_groups(
        paths["datasets"], split_ids["test"], final_stage.feature_groups, final_stage
    )

    if strategy == "residual_correction":
        final_y_train = y_train - base_train_signals[0][1].reshape(-1)
        final_problem_type = "regression"
    else:
        final_problem_type = problem_type
        final_y_train = _target_for_stage(final_problem_type, final_stage, y_train, y_train_labels)

    train_metric_target = _metric_target(problem_type, y_train, y_train_labels)
    test_metric_target = _metric_target(problem_type, y_test_target, y_test_labels)
    meta_columns = _metadata_feature_columns(paths["datasets"], final_groups, x_final_train_meta.shape[1])
    final_variants = _final_feature_variants(
        x_final_train_meta,
        x_final_test_meta,
        meta_columns,
        base_train_signals,
        base_test_signals,
    )
    selected_variant, input_selection_report, selected_input_row = _select_final_input_variant(
        config,
        problem_type,
        final_problem_type,
        final_stage,
        final_variants,
        final_y_train,
        y_train_labels,
        train_metric_target,
        label_values,
    )
    x_final_train = selected_variant["x_train"]
    x_final_test = selected_variant["x_test"]

    final_oof_signal = None
    decision_threshold_report = None
    final_oof_metrics = None
    if selected_input_row:
        final_params = selected_input_row["selected_params"]
        final_cv_results = selected_input_row["cv_results"]
        decision_threshold_report = selected_input_row.get("decision_thresholds")
        final_oof_metrics = selected_input_row.get("oof_metrics")
    else:
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
        if final_problem_type == "classification" and _threshold_settings(final_stage)["enabled"]:
            final_oof_cv = max(
                2,
                final_stage.hyperparameter_search.cv if final_stage.hyperparameter_search.enabled else 5,
            )
            final_oof_signal = _out_of_fold_signal(
                final_problem_type,
                final_stage,
                x_final_train,
                final_y_train,
                final_params,
                config.problem.random_state,
                final_oof_cv,
                split_labels=y_train_labels,
                stage_name="final threshold",
            )
            decision_threshold_report = _tune_decision_threshold(
                final_stage,
                final_oof_signal,
                y_train_labels,
                label_values,
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
        y_pred = _classification_prediction_from_signal(final_signal, decision_threshold_report)
    else:
        y_pred = final_raw_pred

    x_final_train_proc = _transform_with_scaler(x_final_train, final_scaler)
    if strategy == "residual_correction":
        final_train_pred = base_train_signals[0][1].reshape(-1) + final_model.predict(x_final_train_proc)
    elif problem_type == "classification":
        final_train_signal = _predict_signal(problem_type, final_model, x_final_train_proc)
        final_train_pred = _classification_prediction_from_signal(final_train_signal, decision_threshold_report)
    else:
        final_train_pred = final_model.predict(x_final_train_proc)

    metrics = _evaluate(problem_type, test_metric_target, y_pred, label_values)
    final_train_metrics = _evaluate(problem_type, train_metric_target, final_train_pred, label_values)
    if final_oof_signal is not None:
        final_oof_pred = _classification_prediction_from_signal(final_oof_signal, decision_threshold_report)
        final_oof_metrics = _evaluate(problem_type, train_metric_target, final_oof_pred, label_values)
    if decision_threshold_report:
        metrics = {**metrics, "decision_thresholds": decision_threshold_report}
    split_distribution = (
        {"train": _class_distribution(y_train_labels), "test": _class_distribution(y_test_labels)}
        if problem_type == "classification"
        else None
    )
    artifacts: dict[str, Path] = {}
    artifacts.update(
        {f"final_{key}": value for key, value in _save_stage_artifacts(
            paths["models"] / "final", final_model, final_scaler, final_params, final_cv_results
        ).items()}
    )
    if input_selection_report:
        artifacts["final_input_selection_metrics"] = _write_json(
            paths["metrics"] / "final_input_selection_metrics.json",
            input_selection_report,
        )
    artifacts["model"] = artifacts["final_model"]
    base_prediction_columns = []
    extra_prediction_columns = {}
    for base_name, base_test_signal in base_test_signals:
        column_name = f"base_{_stage_slug(base_name)}_signal"
        extra_prediction_columns[column_name] = base_test_signal
        base_prediction_columns.extend(_base_signal_columns(base_name, base_test_signal))
    if problem_type == "classification" and strategy != "residual_correction":
        if final_signal is not None and final_signal.ndim > 1 and final_signal.shape[1] > 1:
            extra_prediction_columns["final_class_probability"] = final_signal
        if np.asarray(y_test_target).ndim > 1:
            extra_prediction_columns["target_probability"] = y_test_target
    final_report = {
        "stage": "final",
        "name": "final",
        "family": _effective_family(final_stage, final_params),
        "feature_groups": final_groups,
        "input_variant": selected_variant["name"],
        "input_feature_count": int(x_final_train.shape[1]),
        "input_feature_columns": selected_variant["feature_columns"],
        "base_prediction_columns": base_prediction_columns,
        "selected_params": final_params,
        "cv_results": final_cv_results,
        "training": _training_info(final_model),
        "train_metrics": final_train_metrics,
        "test_metrics": metrics,
    }
    if final_oof_metrics is not None:
        final_report["train_oof_metrics"] = final_oof_metrics
    if decision_threshold_report:
        final_report["decision_thresholds"] = decision_threshold_report
    if input_selection_report:
        final_report["input_selection"] = {
            "enabled": True,
            "selected_variant": input_selection_report["selected_variant"],
            "selected_score": input_selection_report["selected_score"],
            "scoring": input_selection_report["scoring"],
            "tie_tolerance": input_selection_report["tie_tolerance"],
            "tie_breaker": input_selection_report["tie_breaker"],
        }
    ablation_path = _save_final_ablation_report(
        config,
        problem_type,
        final_problem_type,
        final_stage,
        paths,
        x_final_train_meta,
        x_final_test_meta,
        final_groups,
        base_train_signals,
        base_test_signals,
        final_y_train,
        y_train_labels,
        train_metric_target,
        test_metric_target,
        label_values,
        final_params,
        decision_threshold_report,
    )
    if ablation_path is not None:
        artifacts["final_ablation_metrics"] = ablation_path
    training_report = {
        "strategy": strategy,
        "problem_type": problem_type,
        "class_encoding": config.problem.class_encoding,
        "class_distribution": split_distribution,
        "base_models": {
            name: {
                "feature_groups": base_group_payload.get(name, []),
                "selected_params": base_param_payload.get(name, {}),
            }
            for name, _ in base_train_signals
        },
        "final_model": {
            "family": _effective_family(final_stage, final_params),
            "feature_groups": final_groups,
            "input_variant": selected_variant["name"],
            "input_feature_count": int(x_final_train.shape[1]),
            "input_feature_columns": selected_variant["feature_columns"],
            "base_prediction_columns": base_prediction_columns,
            "decision_thresholds": decision_threshold_report,
            "input_selection": input_selection_report,
        },
        "stages": list(base_reports) + [final_report],
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
                "final_input_variant": selected_variant["name"],
                "final_input_feature_columns": selected_variant["feature_columns"],
                "final_input_selection": input_selection_report,
                "base_prediction_columns": base_prediction_columns,
                "decision_thresholds": decision_threshold_report,
                "class_distribution": split_distribution,
                "class_encoding": config.problem.class_encoding,
            },
            extra_prediction_columns=extra_prediction_columns,
        )
    )
    return artifacts


def train_final_model(config: PipelineConfig, run_root: str | Path, interactive: bool = False) -> dict[str, Path]:
    if config.model.strategy == "direct":
        raise ValueError("Final-stage training from base signals is only available for stacked or residual strategies.")
    paths = _training_paths(run_root)
    context = _prepare_training_inputs(config, paths, interactive=interactive)
    base_reports, base_train_signals, base_test_signals, base_group_payload, base_param_payload = _load_base_reports_and_signals(
        config, paths, context["split_ids"]
    )
    return _train_final_from_base_signals(
        config,
        context["problem_type"],
        paths,
        context["split_ids"],
        context["y_train"],
        context["y_test_target"],
        context["y_train_labels"],
        context["y_test_labels"],
        context["label_values"],
        base_reports,
        base_train_signals,
        base_test_signals,
        base_group_payload,
        base_param_payload,
    )


def train_model(config: PipelineConfig, run_root: str | Path, interactive: bool = False) -> dict[str, Path]:
    paths = _training_paths(run_root)
    context = _prepare_training_inputs(config, paths, interactive=interactive)
    problem_type = context["problem_type"]

    if config.model.strategy == "direct":
        x_train, group_names = _load_matrix_for_groups(
            paths["datasets"], context["split_ids"]["train"], config.model.feature_groups, config.model
        )
        x_test, _ = _load_matrix_for_groups(
            paths["datasets"], context["split_ids"]["test"], config.model.feature_groups, config.model
        )
        return _train_direct(
            config,
            problem_type,
            paths,
            context["split_ids"],
            x_train,
            x_test,
            context["y_train"],
            context["y_test_target"],
            context["y_train_labels"],
            context["y_test_labels"],
            context["label_values"],
            group_names,
        )

    return _train_stacked_or_residual(
        config,
        problem_type,
        paths,
        context["split_ids"],
        context["y_train"],
        context["y_test_target"],
        context["y_train_labels"],
        context["y_test_labels"],
        context["label_values"],
    )
