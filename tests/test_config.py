from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import ModelConfig, PipelineConfig, ProblemConfig, ProductSpec, TargetConfig, load_config
from src.data.preprocessing import transform_target
from src.data.targets import metadata_to_dataarray
from src.models.training import (
    _augment_training_data,
    _candidate_pool,
    _load_matrix_for_groups,
    _make_sample_weights,
    interval_soft_labeling,
    train_model,
)


def test_pseudonitzschia_configs_are_single_model_experiments():
    optics = load_config("configs/pseudonitzschia_optics_classification.yaml")
    environment = load_config("configs/pseudonitzschia_environment_classification.yaml")

    assert optics.model.family == "cnn3d"
    assert optics.model.feature_groups == ["optics"]
    assert [product.name for product in optics.products] == [
        "reflectance",
        "inherent_optics",
        "plankton_l3",
    ]
    assert optics.problem.class_encoding == "soft_probabilities"
    assert optics.problem.target_transform_offset == 100.0
    assert optics.problem.test_size == 0.2

    assert environment.model.family == "cnn3d"
    assert environment.model.feature_groups == ["nut", "car", "phy"]
    assert environment.problem.test_size == 0.2
    assert len(environment.products) == 10
    assert environment.products[0].preprocess["derived_variables"][0]["name"] == "din"
    assert environment.products[7].preprocess["exclude_from_log1p"] == ["ph"]
    assert [candidate["name"] for candidate in environment.model.hyperparameter_search.candidates] == [
        "m",
        "small",
        "small_regularized",
    ]

    for config in [optics, environment]:
        assert not hasattr(config.model, "strategy")
        assert len(config.model.hyperparameter_search.candidates) == 3
        assert config.model.sample_weight["mode"] == "balanced"
        assert config.model.augmentation["repetitions"] == 10


def test_unknown_model_keys_are_rejected():
    data = {
        "target": {"path": "targets.csv", "target_column": "target"},
        "products": [{"name": "x", "dataset_id": "local"}],
        "model": {"unknown_option": True, "family": "random_forest"},
    }

    with pytest.raises(ValueError, match="one direct model"):
        PipelineConfig.from_dict(data)


def test_interval_soft_labeling_returns_smooth_probabilities():
    probs = interval_soft_labeling(
        np.array([0.5, 1.0, 1.5]),
        [[0.0, 1.0], [1.0, 2.0]],
        temperature=0.5,
        prior=0.05,
    )

    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > probs[0, 1]
    assert probs[-1, 1] > probs[-1, 0]


def test_metadata_preserves_missing_values_and_adds_cyclic_day():
    data = pd.DataFrame(
        {
            "Id": [1, 2],
            "time": pd.to_datetime(["2024-01-01", "2024-07-01"]),
            "tem": [15.0, np.nan],
            "sal": [37.0, 38.0],
        }
    )

    meta = metadata_to_dataarray(
        data,
        metadata_columns=["tem", "sal"],
        include_spatial_metadata=False,
        include_day_metadata=False,
        include_cyclic_day_metadata=True,
    )

    assert list(meta.coords["variable"].values) == ["x_day", "y_day", "tem", "sal"]
    assert np.isnan(meta.sel(Id=2, variable="tem").item())


def test_log_target_transform_supports_offset():
    target = xr.DataArray([0.0, 900.0], dims="Id", coords={"Id": [1, 2]})
    transformed = transform_target(target, "log", offset=100.0)

    assert np.allclose(transformed.values, np.log([100.0, 1000.0]))


def test_cnn_loader_fills_missing_values_but_tree_loader_keeps_them(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    data = xr.DataArray(
        np.ones((2, 2, 2, 2, 3), dtype=np.float32),
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": [1, 2],
            "lat": [0, 1],
            "lon": [0, 1],
            "time": [0, 1],
            "variable": ["a", "b", "c"],
        },
    )
    data.values[0, 0, 0, 0, 0] = np.nan
    data.to_netcdf(datasets_dir / "optics.nc")

    cnn_x, _ = _load_matrix_for_groups(
        datasets_dir,
        [1, 2],
        ["optics"],
        ModelConfig(family="cnn3d", feature_groups=["optics"]),
    )
    tree_x, _ = _load_matrix_for_groups(
        datasets_dir,
        [1, 2],
        ["optics"],
        ModelConfig(family="random_forest", feature_groups=["optics"]),
    )

    assert cnn_x.shape == (2, 2, 2, 2, 3)
    assert np.isfinite(cnn_x).all()
    assert tree_x.shape == (2, 24)
    assert np.isnan(tree_x).sum() == 1


def test_candidate_pool_combines_fixed_candidates_and_random_samples():
    model = ModelConfig.from_dict(
        {
            "family": "random_forest",
            "params": {"random_state": 42},
            "hyperparameter_search": {
                "enabled": True,
                "candidates": [{"n_estimators": 10}],
                "param_distributions": {
                    "max_depth": [2, 4],
                    "min_samples_leaf": {"type": "randint", "low": 1, "high": 3},
                },
                "n_iter": 2,
                "random_state": 42,
            },
        }
    )

    candidates = _candidate_pool(model)

    assert len(candidates) == 3
    assert candidates[0]["n_estimators"] == 10
    assert all(candidate["random_state"] == 42 for candidate in candidates)


def test_balanced_weights_and_augmentation_preserve_targets():
    model = ModelConfig(
        family="cnn3d",
        feature_groups=["optics"],
        sample_weight={"mode": "balanced", "class_boost": [1.0, 1.0, 2.0]},
        augmentation={
            "enabled": True,
            "repetitions": 2,
            "seed": 123,
            "noise_std": {"optics": [0.1, 0.0]},
        },
    )
    labels = np.array([0, 0, 1, 2])
    targets = np.eye(3)[labels]
    x = np.ones((4, 2, 2, 2, 2), dtype=np.float32)

    weights, weight_info = _make_sample_weights("classification", model, labels)
    x_fit, y_fit, labels_fit, weights_fit, augmentation_info = _augment_training_data(
        x,
        targets,
        labels,
        weights,
        model,
        random_state=42,
    )

    assert weight_info["class_distribution"] == {"0": 2, "1": 1, "2": 1}
    assert x_fit.shape[0] == 12
    assert y_fit.shape == (12, 3)
    assert labels_fit.tolist() == labels.tolist() + np.repeat(labels, 2).tolist()
    assert weights_fit.shape == (12,)
    assert augmentation_info["fit_samples"] == 12


def test_direct_training_saves_single_model_outputs(tmp_path):
    run_root = tmp_path / "run"
    datasets_dir = run_root / "datasets"
    datasets_dir.mkdir(parents=True)
    ids = np.arange(30)
    labels = np.tile(np.arange(3), 10)
    features = np.column_stack(
        [
            labels == 0,
            labels == 1,
            labels == 2,
            ids / len(ids),
        ]
    ).astype(np.float32)

    xr.DataArray(
        labels.astype(float).reshape(-1, 1),
        dims=("Id", "variable"),
        coords={"Id": ids, "variable": ["target"]},
    ).to_netcdf(datasets_dir / "target.nc")
    xr.DataArray(
        features,
        dims=("Id", "variable"),
        coords={"Id": ids, "variable": ["a", "b", "c", "d"]},
    ).to_netcdf(datasets_dir / "optics.nc")

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[ProductSpec(name="local", dataset_ids=["local"], feature_group="optics")],
        problem=ProblemConfig(
            type="classification",
            class_intervals=[[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]],
            class_encoding="hard",
            test_size=0.3,
            random_state=42,
        ),
        model=ModelConfig(
            family="random_forest",
            feature_groups=["optics"],
            params={"n_estimators": 20, "random_state": 42},
        ),
    )

    artifacts = train_model(config, run_root)
    report = json.loads((run_root / "reports" / "training_report.json").read_text())

    assert artifacts["model"] == run_root / "models" / "model.pkl"
    assert artifacts["selection"] == run_root / "models" / "selection.json"
    assert artifacts["metrics"] == run_root / "metrics" / "metrics.json"
    assert artifacts["training_metrics"] == run_root / "metrics" / "training_metrics.json"
    assert artifacts["predictions"] == run_root / "metrics" / "predictions.csv"
    assert (run_root / "metrics" / "confusion_matrix.jpg").exists()
    assert report["feature_groups"] == ["optics"]
    assert "test_metrics" in report
