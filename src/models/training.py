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
from sklearn.model_selection import KFold, ParameterSampler, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from src.config import ModelConfig, PipelineConfig
from src.metrics.classification import (
    classification_metrics,
    save_confusion_matrix_plot,
)
from src.metrics.regression import regression_metrics
from src.models.factory import REPORT_PARAM_KEYS, create_model
from src.models.tree import save_pickle_model


DISALLOWED_DECISION_PARAM_KEYS = {
    "decision_class_index",
    "decision_threshold",
    "use_decision_threshold",
}


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
    def __init__(self, passthrough_indices: list[int] | None = None):
        self.mean_ = None
        self.std_ = None
        self.passthrough_indices = [] if passthrough_indices is None else list(passthrough_indices)

    def fit(self, x: np.ndarray) -> "ArrayStandardizer":
        axes = tuple(range(x.ndim - 1))
        self.mean_ = np.nanmean(x, axis=axes, keepdims=True)
        self.std_ = np.nanstd(x, axis=axes, keepdims=True)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        if self.passthrough_indices:
            self.mean_[..., self.passthrough_indices] = 0.0
            self.std_[..., self.passthrough_indices] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The array standardizer has not been fitted.")
        transformed = (x - self.mean_) / self.std_
        if self.passthrough_indices:
            transformed[..., self.passthrough_indices] = x[..., self.passthrough_indices]
        return transformed

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def _model_uses_cnn3d(model_config: ModelConfig) -> bool:
    return model_config.family.lower() in {"cnn3d", "3d_cnn"}


def _model_uses_keras(model_config: ModelConfig) -> bool:
    return model_config.family.lower() in {"cnn3d", "3d_cnn", "dense", "mlp", "dense_nn", "tabular_nn"}


def _model_accepts_probability_targets(model_config: ModelConfig) -> bool:
    return model_config.family.lower() in {
        "cnn",
        "cnn1d",
        "1d_cnn",
        "cnn3d",
        "3d_cnn",
        "dense",
        "mlp",
        "dense_nn",
        "tabular_nn",
    }


def _feature_group_paths(datasets_dir: Path, requested_groups: list[str]) -> list[Path]:
    if requested_groups:
        return [datasets_dir / f"{group}.nc" for group in requested_groups]
    return sorted(path for path in datasets_dir.glob("*.nc") if path.stem != "target")


def _load_group(path: Path, ids, flatten: bool = True) -> np.ndarray:
    data = xr.load_dataarray(path).sel(Id=ids)
    if not flatten:
        data = data.transpose("Id", "lat", "lon", "time", "variable")
        return np.asarray(data.values)
    return np.asarray(data.values).reshape(len(ids), -1)


def _group_channel_names(path: Path) -> list[str]:
    data = xr.load_dataarray(path)
    if "variable" not in data.coords:
        return []
    return [str(value) for value in data["variable"].values]


def _load_matrix_for_groups(
    datasets_dir: Path,
    ids,
    feature_groups: list[str],
    model_config: ModelConfig | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    group_paths = _feature_group_paths(datasets_dir, feature_groups)
    missing = [path for path in group_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature group files: {missing}")
    channel_names = [name for path in group_paths for name in _group_channel_names(path)]
    if model_config is not None and _model_uses_cnn3d(model_config):
        x = np.concatenate([_load_group(path, ids, flatten=False) for path in group_paths], axis=-1)
        x = np.where(np.isinf(x), np.nan, x)
    else:
        x = np.hstack([_load_group(path, ids) for path in group_paths])
        x = np.where(np.isinf(x), np.nan, x)
    return x, [path.stem for path in group_paths], channel_names


def _feature_columns(datasets_dir: Path, group_names: list[str], ids, width: int) -> list[str]:
    columns: list[str] = []
    for group_name in group_names:
        path = datasets_dir / f"{group_name}.nc"
        data = xr.load_dataarray(path)
        if data.dims == ("Id", "variable"):
            columns.extend([str(value) for value in data["variable"].values])
        else:
            group_width = int(np.asarray(data.sel(Id=ids[:1]).values).reshape(1, -1).shape[1])
            columns.extend([f"{group_name}_{index}" for index in range(group_width)])
    if len(columns) != width:
        return [f"feature_{index}" for index in range(width)]
    return columns


def _mask_channel_indices(channel_names: list[str]) -> list[int]:
    return [
        index
        for index, name in enumerate(channel_names)
        if str(name).split(":")[-1] in {"cloud_mask", "land_mask"}
    ]


def _product_group_name(product) -> str:
    return product.feature_group or product.preprocess.get("feature_group") or product.name


def _product_channel_specs(product, config: PipelineConfig) -> list[tuple[str, str]]:
    names = [product.rename_variables.get(var, var) for var in product.variables]
    names.extend(spec["name"] for spec in _derived_variable_specs_from_mapping(product.preprocess))
    log_excluded = {str(value) for value in _as_list_for_training(product.preprocess.get("exclude_from_log"))}
    log1p_excluded = {str(value) for value in _as_list_for_training(product.preprocess.get("exclude_from_log1p"))}
    use_log = bool(product.preprocess.get("log", config.preprocess.log_products))
    use_log1p = bool(product.preprocess.get("log1p", False))
    prefix_variables = bool(product.preprocess.get("prefix_variables", config.preprocess.prefix_variables))

    specs = []
    for name in names:
        transform = "none"
        if use_log and name not in log_excluded:
            transform = "log"
        if use_log1p and name not in log1p_excluded:
            transform = "log1p"
        output_name = f"{product.name}:{name}" if prefix_variables else str(name)
        specs.append((output_name, transform))

    add_masks = bool(product.preprocess.get("add_cloud_land_masks", config.preprocess.add_cloud_land_masks))
    if add_masks:
        mask_kinds = {
            str(value)
            for value in _as_list_for_training(product.preprocess.get("mask_kinds", ["cloud_mask", "land_mask"]))
        }
        if "cloud_mask" in mask_kinds:
            specs.append(("cloud_mask", "mask"))
        if "land_mask" in mask_kinds:
            specs.append(("land_mask", "mask"))
    return specs


def _as_list_for_training(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _derived_variable_specs_from_mapping(preprocess: dict[str, Any]) -> list[dict[str, str]]:
    raw = _as_list_for_training(preprocess.get("derived_variables"))
    specs = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            specs.append({"name": str(item["name"]), "expression": str(item.get("expression", ""))})
    return specs


def _channel_noise_transforms(config: PipelineConfig, group_names: list[str], channel_names: list[str]) -> list[str]:
    transforms = []
    for group_name in group_names:
        seen_masks = set()
        for product in config.products:
            if _product_group_name(product) != group_name:
                continue
            for name, transform in _product_channel_specs(product, config):
                if transform == "mask":
                    if name in seen_masks:
                        continue
                    seen_masks.add(name)
                transforms.append(transform)
    if len(transforms) != len(channel_names):
        return ["mask" if index in _mask_channel_indices(channel_names) else "none" for index in range(len(channel_names))]
    return transforms


def _prepare_model_target(
    config: PipelineConfig,
    problem_type: str,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    values = np.asarray(values).reshape(-1)
    if problem_type == "regression":
        return values, None, []

    label_encoder = LabelEncoder()
    if config.problem.class_intervals:
        hard_labels = interval_labeling(values.astype(float), config.problem.class_intervals)
        label_values = list(range(len(config.problem.class_intervals)))
        if config.problem.class_encoding == "one_hot":
            return _one_hot(hard_labels, len(label_values)), hard_labels, label_values
        if config.problem.class_encoding == "soft_probabilities":
            soft_labels = interval_soft_labeling(
                values.astype(float),
                config.problem.class_intervals,
                temperature=config.problem.soft_label_temperature,
                prior=config.problem.soft_label_prior,
            )
            return soft_labels, hard_labels, label_values
        return hard_labels, hard_labels, label_values

    hard_labels = label_encoder.fit_transform(values)
    label_values = list(label_encoder.transform(label_encoder.classes_))
    if config.problem.class_encoding == "one_hot":
        return _one_hot(hard_labels, len(label_values)), hard_labels, label_values
    if config.problem.class_encoding == "soft_probabilities":
        raise ValueError("Soft probability labels require problem.class_intervals.")
    return hard_labels, hard_labels, label_values


def _target_for_model(
    problem_type: str,
    model_config: ModelConfig,
    y_target: np.ndarray,
    y_labels: np.ndarray | None,
) -> np.ndarray:
    if (
        problem_type == "classification"
        and np.asarray(y_target).ndim > 1
        and not _model_accepts_probability_targets(model_config)
    ):
        if y_labels is None:
            raise ValueError("Hard labels are required for this classifier.")
        return y_labels
    return y_target


def _class_distribution(labels: np.ndarray | None) -> dict[str, int] | None:
    if labels is None:
        return None
    labels = np.asarray(labels).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    return {str(cls.item() if hasattr(cls, "item") else cls): int(count) for cls, count in zip(classes, counts)}


def _parameter_distribution(spec: Any):
    if not isinstance(spec, dict):
        return spec
    if "values" in spec:
        return list(spec["values"])

    dist_type = str(spec.get("type", spec.get("distribution", ""))).lower()
    if dist_type in {"choice", "categorical"}:
        return list(spec["options"])

    if dist_type in {"randint", "int", "integer"}:
        from scipy.stats import randint

        low = int(spec.get("low", spec.get("min")))
        high = int(spec.get("high", spec.get("max")))
        return randint(low, high + 1)

    if dist_type in {"uniform", "float"}:
        from scipy.stats import uniform

        low = float(spec.get("low", spec.get("min")))
        high = float(spec.get("high", spec.get("max")))
        return uniform(low, high - low)

    if dist_type in {"loguniform", "log_uniform"}:
        from scipy.stats import loguniform

        low = float(spec.get("low", spec.get("min")))
        high = float(spec.get("high", spec.get("max")))
        return loguniform(low, high)

    return spec


def _sampled_candidates(model_config: ModelConfig) -> list[dict[str, Any]]:
    search = model_config.hyperparameter_search
    if not search.param_distributions:
        return []
    distributions = {
        key: _parameter_distribution(value)
        for key, value in search.param_distributions.items()
    }
    sampler = ParameterSampler(
        distributions,
        n_iter=int(search.n_iter),
        random_state=search.random_state,
    )
    return [{**model_config.params, **dict(candidate)} for candidate in sampler]


def _candidate_pool(model_config: ModelConfig) -> list[dict[str, Any]]:
    search = model_config.hyperparameter_search
    if not search.enabled:
        candidates = [dict(model_config.params)]
    else:
        candidates = [{**model_config.params, **candidate} for candidate in search.candidates]
        if search.param_grid:
            keys = list(search.param_grid.keys())
            for values in product(*[search.param_grid[key] for key in keys]):
                candidates.append({**model_config.params, **dict(zip(keys, values))})
        candidates.extend(_sampled_candidates(model_config))
        candidates = candidates or [dict(model_config.params)]
    invalid_keys = sorted({key for candidate in candidates for key in candidate if key in DISALLOWED_DECISION_PARAM_KEYS})
    if invalid_keys:
        raise ValueError(
            "Decision-threshold hyperparameters are no longer supported in model selection: "
            f"{invalid_keys}"
        )
    return candidates


def _default_scoring(problem_type: str) -> str:
    return "f1_macro" if problem_type == "classification" else "r2"


def _splitter(problem_type: str, cv: int, random_state: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def _sample_weight_settings(model_config: ModelConfig) -> dict[str, Any]:
    raw = model_config.sample_weight
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
    raise ValueError("model.sample_weight must be false, 'balanced', or a mapping.")


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
    model_config: ModelConfig,
    labels: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    settings = _sample_weight_settings(model_config)
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


def _noise_std_array(model_config: ModelConfig, x: np.ndarray) -> tuple[np.ndarray | float, dict[str, Any]]:
    config = model_config.augmentation
    std = config.get("noise_std", config.get("std", config.get("std_x", 0.0)))
    if isinstance(std, dict):
        values: list[Any] = []
        for group in model_config.feature_groups:
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


def _zero_mask_noise(noise_std: np.ndarray | float, mask_channel_indices: list[int]) -> np.ndarray | float:
    if not mask_channel_indices or np.isscalar(noise_std):
        return noise_std
    adjusted = np.array(noise_std, copy=True)
    adjusted[..., mask_channel_indices] = 0.0
    return adjusted


def _noise_for_channel(noise_std: np.ndarray | float, channel: int) -> float:
    if np.isscalar(noise_std):
        return float(noise_std)
    return float(np.asarray(noise_std).reshape(-1)[channel])


def _apply_feature_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    noise_std: np.ndarray | float,
    channel_noise_transforms: list[str] | None,
) -> np.ndarray:
    transforms = channel_noise_transforms or ["none"] * x.shape[-1]
    if len(transforms) != x.shape[-1]:
        transforms = ["none"] * x.shape[-1]
    x_aug = x.astype(np.float32, copy=True)
    for channel, transform in enumerate(transforms):
        std = _noise_for_channel(noise_std, channel)
        if std == 0.0 or transform == "mask":
            continue
        values = x_aug[..., channel]
        if transform == "log":
            x_aug[..., channel] = values + rng.normal(0.0, std, size=values.shape)
        elif transform == "log1p":
            raw = np.maximum(np.expm1(values), 0.0)
            raw = raw * np.exp(rng.normal(0.0, std, size=values.shape))
            x_aug[..., channel] = np.log1p(raw)
        else:
            x_aug[..., channel] = values + rng.normal(0.0, std * np.abs(values), size=values.shape)
    return x_aug


def _augment_training_data(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None,
    sample_weight: np.ndarray | None,
    model_config: ModelConfig,
    random_state: int,
    mask_channel_indices: list[int] | None = None,
    channel_noise_transforms: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    config = model_config.augmentation
    if not config or not bool(config.get("enabled", config.get("augment", False))):
        return x, y, labels, sample_weight, {"enabled": False, "input_samples": int(len(x)), "fit_samples": int(len(x))}

    repetitions = int(config.get("repetitions", 1))
    if repetitions <= 0:
        return x, y, labels, sample_weight, {"enabled": False, "input_samples": int(len(x)), "fit_samples": int(len(x))}

    rng = np.random.default_rng(int(config.get("seed", random_state)))
    x_aug = np.repeat(x, repetitions, axis=0).astype(np.float32, copy=True)
    noise_std, noise_info = _noise_std_array(model_config, x_aug)
    mask_channel_indices = [] if mask_channel_indices is None else list(mask_channel_indices)
    if mask_channel_indices:
        noise_std = _zero_mask_noise(noise_std, mask_channel_indices)
        noise_info = {**noise_info, "mask_channels_without_noise": mask_channel_indices}
    if np.any(np.asarray(noise_std) != 0):
        x_aug = _apply_feature_noise(x_aug, rng, noise_std, channel_noise_transforms)
    if mask_channel_indices:
        x_aug[..., mask_channel_indices] = np.repeat(x, repetitions, axis=0)[..., mask_channel_indices]
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
        "noise_semantics": "pre_standardization_log_space_or_relative",
    }


def _fit_scaler_if_needed(
    x: np.ndarray,
    model_config: ModelConfig,
    mask_channel_indices: list[int] | None = None,
) -> tuple[np.ndarray, StandardScaler | ArrayStandardizer | None]:
    if not model_config.standardize:
        return x, None
    if _model_uses_cnn3d(model_config):
        scaler = ArrayStandardizer(passthrough_indices=mask_channel_indices)
        return scaler.fit_transform(x), scaler
    scaler = StandardScaler()
    return scaler.fit_transform(x), scaler


def _fill_missing_for_model(x: np.ndarray, model_config: ModelConfig) -> np.ndarray:
    if _model_uses_cnn3d(model_config):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _transform_with_scaler(
    x: np.ndarray,
    scaler: StandardScaler | ArrayStandardizer | None,
    model_config: ModelConfig,
) -> np.ndarray:
    transformed = x if scaler is None else scaler.transform(x)
    transformed = _fill_missing_for_model(transformed, model_config)
    passthrough = getattr(scaler, "passthrough_indices", []) if scaler is not None else []
    if passthrough:
        transformed[..., passthrough] = (transformed[..., passthrough] > 0.5).astype(np.float32)
    return transformed


def _estimator_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key not in REPORT_PARAM_KEYS}


def _prediction_rule_name(problem_type: str, params: dict[str, Any]) -> str:
    if problem_type != "classification":
        return "direct"
    return "argmax"


def _validate_target_compatibility(
    problem_type: str,
    model_config: ModelConfig,
    y: np.ndarray,
) -> None:
    if problem_type != "classification":
        return
    if np.asarray(y).ndim <= 1:
        return
    if not _model_accepts_probability_targets(model_config):
        raise ValueError(
            "Probability-vector classification targets require a model family that accepts "
            "2D class targets, such as cnn3d or dense. Use problem.class_encoding: hard for tree models."
        )


def _fit_estimator(
    problem_type: str,
    model_config: ModelConfig,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    y_labels: np.ndarray | None = None,
    random_state: int = 42,
    progress_description: str | None = None,
    mask_channel_indices: list[int] | None = None,
    channel_noise_transforms: list[str] | None = None,
):
    _validate_target_compatibility(problem_type, model_config, y)
    sample_weight, sample_weight_info = _make_sample_weights(problem_type, model_config, y_labels)
    x_fit_raw, y_fit, labels_fit, sample_weight_fit, augmentation_info = _augment_training_data(
        x,
        y,
        y_labels,
        sample_weight,
        model_config,
        random_state,
        mask_channel_indices=mask_channel_indices,
        channel_noise_transforms=channel_noise_transforms,
    )
    _, scaler = _fit_scaler_if_needed(x, model_config, mask_channel_indices=mask_channel_indices)
    x_fit = _transform_with_scaler(x_fit_raw, scaler, model_config)
    fit_params = dict(params)
    if progress_description and _model_uses_keras(model_config):
        fit_params.setdefault("progress_description", progress_description)
    model = create_model(problem_type, model_config, params=_estimator_params(fit_params))
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
        "class_distribution": _class_distribution(labels_fit),
        "sample_weight": sample_weight_info,
        "augmentation": augmentation_info,
    }
    return model, scaler


def _score_classification_predictions(y_true, y_pred, scoring: str) -> float:
    metrics = classification_metrics(y_true, y_pred)
    scoring = str(scoring).lower()
    if scoring == "accuracy":
        return float(metrics["accuracy"])
    if scoring in {"f1", "f1_macro"}:
        return float(metrics["f1_macro"])
    if scoring == "precision_macro":
        return float(metrics["precision_macro"])
    if scoring == "recall_macro":
        return float(metrics["recall_macro"])
    raise ValueError(
        "Classification CV scoring supports accuracy, f1_macro, "
        "precision_macro, and recall_macro."
    )


def _epochs_ran(model) -> int | None:
    history = _history_summary(model)
    if not history:
        return None
    epochs = history.get("epochs_ran")
    return None if epochs is None else int(epochs)


def _select_params(
    problem_type: str,
    model_config: ModelConfig,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    split_labels: np.ndarray | None = None,
    score_labels: np.ndarray | None = None,
    mask_channel_indices: list[int] | None = None,
    channel_noise_transforms: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = model_config.hyperparameter_search
    candidates = _candidate_pool(model_config)
    if not search.enabled or len(candidates) == 1:
        return candidates[0], []

    scorer = get_scorer(search.scoring or _default_scoring(problem_type))
    splitter = _splitter(problem_type, search.cv, random_state)
    split_y = y if split_labels is None else split_labels
    score_y = y if score_labels is None else score_labels
    cv_results = []
    candidate_bar = tqdm(candidates, desc="model hyperparameters", unit="candidate")
    for index, params in enumerate(candidate_bar):
        scores = []
        fold_epochs = []
        splits = list(splitter.split(x, split_y))
        fold_bar = tqdm(
            splits,
            desc=f"candidate {index + 1}/{len(candidates)} CV",
            unit="fold",
            leave=False,
        )
        for fold_index, (train_idx, val_idx) in enumerate(fold_bar, start=1):
            train_labels = split_y[train_idx] if problem_type == "classification" else None
            model, scaler = _fit_estimator(
                problem_type,
                model_config,
                x[train_idx],
                y[train_idx],
                params,
                y_labels=train_labels,
                random_state=random_state + index,
                progress_description=f"candidate {index + 1}/{len(candidates)} fold {fold_index}/{len(splits)}",
                mask_channel_indices=mask_channel_indices,
                channel_noise_transforms=channel_noise_transforms,
            )
            epochs = _epochs_ran(model)
            if epochs is not None:
                fold_epochs.append(epochs)
            x_val = _transform_with_scaler(x[val_idx], scaler, model_config)
            if problem_type == "classification":
                val_signal = _predict_signal(problem_type, model, x_val)
                val_pred = _prediction_labels(problem_type, val_signal)
                score = _score_classification_predictions(
                    score_y[val_idx],
                    val_pred,
                    search.scoring or _default_scoring(problem_type),
                )
            else:
                score = float(scorer(model, x_val, score_y[val_idx]))
            scores.append(score)
            postfix = {"score": f"{score:.4g}"}
            if epochs is not None:
                postfix["epochs"] = str(epochs)
            fold_bar.set_postfix(**postfix)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        epoch_summary = {}
        if fold_epochs:
            epoch_summary = {
                "fold_epochs": fold_epochs,
                "mean_epochs": float(np.mean(fold_epochs)),
                "median_epochs": float(np.median(fold_epochs)),
            }
        cv_results.append(
            {
                "candidate_index": index,
                "params": params,
                "cv": int(search.cv),
                "scores": scores,
                "mean_score": mean_score,
                "std_score": std_score,
                **epoch_summary,
            }
        )
        candidate_bar.set_postfix(best=f"{max(row['mean_score'] for row in cv_results):.4g}")
    best = max(cv_results, key=lambda row: row["mean_score"])
    selected = dict(best["params"])
    if best.get("fold_epochs"):
        median_epochs = float(best["median_epochs"])
        final_epochs = max(1, int(round(median_epochs)))
        selected.update(
            {
                "cv_epochs": list(best["fold_epochs"]),
                "cv_epochs_mean": float(best["mean_epochs"]),
                "cv_epochs_median": median_epochs,
                "epochs": final_epochs,
                "validation_split": 0.0,
                "patience": 0,
                "final_training_epoch_source": "cv_median_epochs_no_validation",
            }
        )
    return selected, cv_results


def _predict_signal(problem_type: str, model, x: np.ndarray) -> np.ndarray:
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))
    prediction = np.asarray(model.predict(x))
    if problem_type == "classification" and prediction.ndim > 1:
        return prediction
    return prediction.reshape(-1, 1)


def _prediction_labels(problem_type: str, prediction: np.ndarray) -> np.ndarray:
    if problem_type == "classification":
        prediction = np.asarray(prediction)
        if prediction.ndim > 1 and prediction.shape[1] > 1:
            return np.argmax(prediction, axis=1)
    return np.asarray(prediction).reshape(-1)


def _metric_target(problem_type: str, y_target: np.ndarray, y_labels: np.ndarray | None) -> np.ndarray:
    return y_labels if problem_type == "classification" else y_target


def _evaluate(problem_type: str, y_true, y_pred, label_values):
    if problem_type == "classification":
        return classification_metrics(y_true, y_pred, labels=label_values)
    return regression_metrics(y_true, y_pred)


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
        summary[f"last_{key}"] = values[-1]
        lower_is_better = any(token in key.lower() for token in ["loss", "error", "mse", "mae"])
        summary[f"best_{key}"] = min(values) if lower_is_better else max(values)
    return summary


def _training_info(model) -> dict[str, Any]:
    info = dict(getattr(model, "_retrieval_training_info", {}))
    history = _history_summary(model)
    if history:
        info["history"] = history
    return info


def _run_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    paths = {
        "root": root,
        "datasets": root / "datasets",
        "models": root / "models",
        "metrics": root / "metrics",
        "reports": root / "reports",
    }
    for path in paths.values():
        if path != root:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_target_values(datasets_dir: Path):
    target = xr.load_dataarray(datasets_dir / "target.nc")
    ids = target.Id.values
    values = np.asarray(target.values).reshape(len(ids), -1)
    if values.shape[1] == 1:
        values = values.reshape(-1)
    return ids, values


def _split_data(x, ids, y_target, y_labels, problem_type: str, test_size: float, random_state: int):
    stratify = y_labels if problem_type == "classification" else None
    arrays = [x, ids, y_target]
    if y_labels is not None:
        arrays.append(y_labels)
    split = train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if y_labels is None:
        x_train, x_test, id_train, id_test, y_train, y_test = split
        return x_train, x_test, id_train, id_test, y_train, y_test, None, None
    x_train, x_test, id_train, id_test, y_train, y_test, label_train, label_test = split
    return x_train, x_test, id_train, id_test, y_train, y_test, label_train, label_test


def _save_predictions(
    path: Path,
    ids,
    problem_type: str,
    y_true,
    y_pred,
    signal: np.ndarray | None = None,
    target_probabilities: np.ndarray | None = None,
    extra_columns: dict[str, Any] | None = None,
) -> Path:
    frame = pd.DataFrame(
        {
            "Id": ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "problem_type": problem_type,
        }
    )
    if problem_type == "classification" and signal is not None and signal.ndim > 1 and signal.shape[1] > 1:
        for index in range(signal.shape[1]):
            frame[f"class_probability_{index}"] = signal[:, index]
    if target_probabilities is not None and np.asarray(target_probabilities).ndim > 1:
        for index in range(target_probabilities.shape[1]):
            frame[f"target_probability_{index}"] = target_probabilities[:, index]
    for column, values in (extra_columns or {}).items():
        frame[column] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _save_confusion_plots(paths: dict[str, Path], y_true, y_pred, labels) -> dict[str, Path]:
    raw_path = paths["metrics"] / "confusion_matrix.jpg"
    normalized_path = paths["metrics"] / "confusion_matrix_normalized_true.jpg"
    save_confusion_matrix_plot(
        y_true,
        y_pred,
        raw_path,
        labels=labels,
        title="Confusion matrix",
    )
    save_confusion_matrix_plot(
        y_true,
        y_pred,
        normalized_path,
        labels=labels,
        normalize="true",
        title="Confusion matrix normalized by true label",
    )
    return {
        "confusion_matrix": raw_path,
        "confusion_matrix_normalized_true": normalized_path,
    }


def train_model(config: PipelineConfig, run_root: str | Path, interactive: bool = False) -> dict[str, Path]:
    problem_type = resolve_problem_type(config, interactive=interactive)
    paths = _run_paths(run_root)
    ids, raw_target = _load_target_values(paths["datasets"])
    y_target, y_labels, label_values = _prepare_model_target(config, problem_type, raw_target)
    x, group_names, channel_names = _load_matrix_for_groups(
        paths["datasets"],
        ids,
        config.model.feature_groups,
        config.model,
    )
    mask_channel_indices = _mask_channel_indices(channel_names)
    channel_noise_transforms = _channel_noise_transforms(config, group_names, channel_names)
    feature_columns = _feature_columns(paths["datasets"], group_names, ids, x.reshape(len(ids), -1).shape[1])

    x_train, x_test, id_train, id_test, y_train, y_test, label_train, label_test = _split_data(
        x,
        ids,
        y_target,
        y_labels,
        problem_type,
        config.problem.test_size,
        config.problem.random_state,
    )
    train_metric_target = _metric_target(problem_type, y_train, label_train)
    test_metric_target = _metric_target(problem_type, y_test, label_test)
    model_y_train = _target_for_model(problem_type, config.model, y_train, label_train)

    selected_params, cv_results = _select_params(
        problem_type,
        config.model,
        x_train,
        model_y_train,
        config.problem.random_state,
        split_labels=label_train if problem_type == "classification" else None,
        score_labels=label_train if problem_type == "classification" else None,
        mask_channel_indices=mask_channel_indices,
        channel_noise_transforms=channel_noise_transforms,
    )
    model, scaler = _fit_estimator(
        problem_type,
        config.model,
        x_train,
        model_y_train,
        selected_params,
        y_labels=label_train,
        random_state=config.problem.random_state,
        progress_description="model fit",
        mask_channel_indices=mask_channel_indices,
        channel_noise_transforms=channel_noise_transforms,
    )
    x_train_proc = _transform_with_scaler(x_train, scaler, config.model)
    x_test_proc = _transform_with_scaler(x_test, scaler, config.model)
    train_signal = _predict_signal(problem_type, model, x_train_proc)
    test_signal = _predict_signal(problem_type, model, x_test_proc)
    train_pred = _prediction_labels(problem_type, train_signal)
    test_pred = _prediction_labels(problem_type, test_signal)
    prediction_rule = _prediction_rule_name(problem_type, selected_params)

    train_metrics = _evaluate(problem_type, train_metric_target, train_pred, label_values)
    test_metrics = _evaluate(problem_type, test_metric_target, test_pred, label_values)

    artifacts: dict[str, Path] = {}
    artifacts["model"] = _save_model_artifact(model, paths["models"] / "model.pkl")
    if scaler is not None:
        with (paths["models"] / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        artifacts["scaler"] = paths["models"] / "scaler.pkl"
    artifacts["selection"] = _write_json(
        paths["models"] / "selection.json",
        {"selected_params": selected_params, "cv_results": cv_results},
    )
    history = _history_payload(model)
    if history:
        artifacts["history"] = _write_json(paths["models"] / "history.json", history)

    plot_artifacts = {}
    if problem_type == "classification":
        plot_artifacts = _save_confusion_plots(paths, test_metric_target, test_pred, label_values)
        artifacts.update(plot_artifacts)

    report = {
        "problem_type": problem_type,
        "class_encoding": config.problem.class_encoding,
        "feature_groups": group_names,
        "input_shape": list(x.shape),
        "input_feature_count": int(x.reshape(len(ids), -1).shape[1]),
        "input_feature_columns": feature_columns,
        "input_channel_names": channel_names,
        "mask_channel_indices": mask_channel_indices,
        "channel_noise_transforms": channel_noise_transforms,
        "selected_params": selected_params,
        "cv_results": cv_results,
        "training": _training_info(model),
        "class_distribution": {
            "train": _class_distribution(label_train),
            "test": _class_distribution(label_test),
        } if problem_type == "classification" else None,
        "prediction_rule": prediction_rule,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix_plots": {key: str(value) for key, value in plot_artifacts.items()},
    }
    artifacts["metrics"] = _write_json(paths["metrics"] / "metrics.json", test_metrics)
    artifacts["training_metrics"] = _write_json(paths["metrics"] / "training_metrics.json", report)
    artifacts["training_report"] = _write_json(paths["reports"] / "training_report.json", report)
    artifacts["split"] = _write_json(
        paths["reports"] / "split.json",
        {
            "train_ids": list(map(str, id_train)),
            "test_ids": list(map(str, id_test)),
            "feature_groups": group_names,
            "selected_params": selected_params,
            "class_distribution": report["class_distribution"],
            "class_encoding": config.problem.class_encoding,
        },
    )
    artifacts["predictions"] = _save_predictions(
        paths["metrics"] / "predictions.csv",
        id_test,
        problem_type,
        test_metric_target,
        test_pred,
        signal=test_signal,
        target_probabilities=y_test if problem_type == "classification" else None,
        extra_columns={
            "prediction_rule": prediction_rule,
        } if problem_type == "classification" else None,
    )
    return artifacts
