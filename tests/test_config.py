import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import (
    ModelConfig,
    ModelStageConfig,
    PipelineConfig,
    PreprocessConfig,
    ProblemConfig,
    ProductSpec,
    TargetConfig,
    load_config,
)
from src.models.training import interval_soft_labeling
from src.models.training import _load_matrix_for_groups
from src.models.training import _target_for_stage
from src.models.training import _augment_training_data
from src.models.training import _make_sample_weights
from src.models.training import _splitter
from src.models.training import train_final_model
from src.models.training import train_model
from src.models.cnn import KerasCNN3DEstimator
from src.metrics.classification import save_confusion_matrix_plot
from src.paths import RunPaths
from src.pipeline.download import download_products
from src.pipeline.matchup import create_matchups
from src.state import PipelineState
from src.data.preprocessing import preprocess_matchups, transform_target
from src.data.targets import metadata_to_dataarray
from src.features.transforms import positive_quantile


def test_load_example_regression_config():
    config = load_config(Path("configs/example_regression.yaml"))
    assert config.problem.type == "regression"
    assert config.target.target_column == "target_value"
    assert config.products[0].name == "reflectance"
    assert config.model.strategy == "direct"
    assert config.model.hyperparameter_search.enabled is True


def test_load_synthetic_end_to_end_config():
    config = load_config(Path("configs/synthetic_end_to_end.yaml"))
    assert config.run_name == "synthetic_end_to_end"
    assert config.run_version == "v1"
    assert config.products[0].source == "local"
    assert config.model.strategy == "stacking"
    assert config.model.base_model.feature_groups == ["optics", "phy"]
    assert config.model.final_model.feature_groups == ["meta"]


def test_load_pseudonitzschia_cnn_classification_config():
    config = load_config(Path("configs/pseudonitzschia_cnn_classification.yaml"))
    assert config.problem.type == "classification"
    assert config.problem.class_encoding == "soft_probabilities"
    assert config.problem.soft_label_temperature == 10.0
    assert config.problem.target_transform_offset == 100.0
    assert np.allclose(
        np.asarray(config.problem.class_intervals)[:, 0],
        np.log([100.0, 1000.0, 100000.0]),
    )
    assert config.target.metadata_columns == ["tem", "sal", "o_perc", "o", "ph", "lat", "lon"]
    assert config.target.include_spatial_metadata is False
    assert config.target.include_day_metadata is False
    assert config.target.include_cyclic_day_metadata is True
    assert config.matchup.time_window_days == 14
    assert config.model.strategy == "stacking"
    assert sorted(config.model.base_models) == ["environment", "optics"]
    assert config.model.base_models["optics"].family == "cnn3d"
    assert config.model.base_models["optics"].feature_groups == ["optics"]
    assert config.model.base_models["environment"].family == "cnn3d"
    assert config.model.base_models["environment"].feature_groups == ["nut", "car", "phy"]
    assert config.model.base_models["optics"].sample_weight["mode"] == "balanced"
    assert config.model.base_models["optics"].augmentation["enabled"] is True
    assert config.model.base_models["optics"].augmentation["repetitions"] == 10
    assert config.model.base_models["environment"].sample_weight["mode"] == "balanced"
    assert config.model.base_models["environment"].augmentation["enabled"] is True
    assert config.model.base_models["environment"].augmentation["repetitions"] == 10
    environment_noise = config.model.base_models["environment"].augmentation["noise_std"]
    assert sum(len(environment_noise[group]) for group in ["nut", "car", "phy"]) == 26
    nutrient_preprocess = next(product.preprocess for product in config.products if product.name == "nutrients")
    assert [spec["name"] for spec in nutrient_preprocess["derived_variables"]] == [
        "din",
        "n_div_p",
        "din_div_p",
        "nh4_div_no3",
    ]
    assert config.model.base_model is None
    assert config.model.final_model.family == "random_forest"
    assert config.model.final_model.feature_groups == ["meta"]
    assert config.model.final_model.params["class_weight"] == "balanced"
    assert config.model.final_model.hyperparameter_search.enabled is False
    assert len(config.model.base_models["optics"].hyperparameter_search.candidates) == 5
    assert len(config.model.base_models["environment"].hyperparameter_search.candidates) == 5
    assert config.products[0].name == "reflectance"
    assert config.products[0].preprocess["mask_kinds"] == ["cloud_mask", "land_mask"]
    assert config.products[0].preprocess["min_valid_ratio"] == 0.3
    assert config.products[3].matchup["lat_window"] == 0.1
    assert config.products[3].matchup["require_full_time_window"] is True


def test_interval_soft_labeling_returns_smooth_probabilities():
    probs = interval_soft_labeling(
        np.array([0.5, 5.0, 10.0]),
        [[0.0, 1.0], [4.0, 6.0], [9.0, 11.0]],
        temperature=1.0,
        prior=0.0,
    )
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.argmax(axis=1).tolist() == [0, 1, 2]


def test_tree_stage_uses_hard_labels_when_pipeline_targets_are_soft():
    soft = np.array([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]])
    hard = np.array([1, 0])

    tree_target = _target_for_stage("classification", ModelStageConfig(family="random_forest"), soft, hard)
    cnn_target = _target_for_stage("classification", ModelStageConfig(family="cnn3d"), soft, hard)

    assert tree_target.tolist() == [1, 0]
    assert np.allclose(cnn_target, soft)


def test_splitter_supports_stratified_cv():
    splitter = _splitter("classification", cv=3, random_state=42)
    x = np.zeros((12, 2))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    splits = list(splitter.split(x, y))

    assert len(splits) == 3


def test_confusion_matrix_plot_is_saved(tmp_path):
    path = tmp_path / "confusion_matrix_normalized_true.jpg"

    save_confusion_matrix_plot([0, 0, 1, 1], [0, 1, 1, 1], path, normalize="true")

    assert path.exists()
    assert path.stat().st_size > 0


def test_positive_quantile_replaces_zero_with_finite_log_floor():
    data = xr.DataArray(
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        dims=["time"],
    )
    floored = positive_quantile(data, quantile=0.01, dims=("time",))

    assert float(floored.min()) > 0
    assert np.isfinite(np.log(floored.values)).all()


def test_target_log_transform_rejects_zero_values():
    target = xr.DataArray(
        np.array([[0.0], [10.0]], dtype=np.float32),
        dims=("Id", "variable"),
        coords={"Id": [1, 2], "variable": ["target"]},
    )

    with pytest.raises(ValueError, match="target_transform: log"):
        transform_target(target, "log")


def test_target_log_transform_accepts_offset_for_zero_values():
    target = xr.DataArray(
        np.array([[0.0], [10.0]], dtype=np.float32),
        dims=("Id", "variable"),
        coords={"Id": [1, 2], "variable": ["target"]},
    )

    transformed = transform_target(target, "log", offset=100.0)

    assert np.allclose(transformed.values.reshape(-1), np.log([100.0, 110.0]))


def test_metadata_to_dataarray_can_use_only_columns_and_cyclic_day():
    data = pd.DataFrame(
        {
            "Id": [1],
            "lat": [40.0],
            "lon": [1.0],
            "time": [pd.Timestamp("2020-04-01")],
            "target": [1.0],
            "tem": [18.0],
            "sal": [37.0],
            "o_perc": [95.0],
            "o": [8.0],
            "ph": [8.1],
        }
    )

    meta = metadata_to_dataarray(
        data,
        ["tem", "sal", "o_perc", "o", "ph"],
        include_spatial_metadata=False,
        include_day_metadata=False,
        include_cyclic_day_metadata=True,
    )

    assert meta["variable"].values.tolist() == ["x_day", "y_day", "tem", "sal", "o_perc", "o", "ph"]


def test_remote_download_writes_marker_without_materializing_dataset(tmp_path):
    product = ProductSpec(
        name="remote_optics",
        dataset_ids=["copernicus-dataset"],
        source="copernicus",
        variables=["RRS443"],
    )
    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "targets.csv"), target_column="target"),
        products=[product],
    )
    paths = RunPaths(tmp_path / "run").ensure()
    state = PipelineState(paths.state_file)

    artifacts = download_products(config, paths, state)

    marker_path = paths.raw / "remote_optics.remote.json"
    raw_path = paths.raw / "remote_optics.nc"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert artifacts["remote_optics"] == marker_path
    assert marker["dataset_ids"] == ["copernicus-dataset"]
    assert "opened lazily" in marker["note"]
    assert not raw_path.exists()


def test_matchups_ignore_stale_raw_files_for_remote_products(tmp_path, monkeypatch):
    product = ProductSpec(
        name="remote_optics",
        dataset_ids=["copernicus-dataset"],
        source="copernicus",
        variables=["RRS443"],
    )
    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "targets.csv"), target_column="target"),
        products=[product],
    )
    paths = RunPaths(tmp_path / "run").ensure()
    state = PipelineState(paths.state_file)
    (paths.raw / "remote_optics.nc").write_text("stale raw data", encoding="utf-8")
    observations = pd.DataFrame(
        {
            "Id": [1],
            "lat": [40.0],
            "lon": [1.0],
            "time": [pd.Timestamp("2020-01-01")],
            "target": [1.0],
        }
    )
    captured = {}

    monkeypatch.setattr(
        "src.pipeline.matchup.load_target_table",
        lambda target: observations,
    )

    def fake_create_product_matchups(product, targets, matchup, raw_path=None):
        captured["raw_path"] = raw_path
        return None, targets[["Id"]].copy()

    monkeypatch.setattr(
        "src.pipeline.matchup.create_product_matchups",
        fake_create_product_matchups,
    )

    create_matchups(config, paths, state)

    assert captured["raw_path"] is None


def test_preprocess_combines_products_with_different_absolute_time_coords(tmp_path):
    paths = RunPaths(tmp_path / "run").ensure()
    targets = pd.DataFrame(
        {
            "Id": [1],
            "lat": [40.0],
            "lon": [1.0],
            "time": [pd.Timestamp("2020-01-15")],
            "target": [1.0],
        }
    )
    targets.to_csv(paths.processed / "targets.csv", index=False)

    def write_matchup(product_name: str, variable_name: str, start_date: str) -> None:
        values = np.ones((1, 2, 2, 2), dtype=np.float32)
        dates = pd.date_range(start_date, periods=2)
        ds = xr.Dataset(
            {variable_name: (("Id", "lat", "lon", "time"), values)},
            coords={"Id": [1], "lat": [0, 1], "lon": [0, 1], "time": [0, 1]},
        )
        ds = ds.assign_coords(
            {
                "lat": xr.DataArray([[40.0, 40.01]], dims=["Id", "lat"]),
                "lon": xr.DataArray([[1.0, 1.01]], dims=["Id", "lon"]),
                "time": xr.DataArray([dates.values], dims=["Id", "time"]),
            }
        )
        ds.to_netcdf(paths.matchups / f"{product_name}.nc")

    write_matchup("product_a", "a", "2020-01-01")
    write_matchup("product_b", "b", "2020-01-02")

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[
            ProductSpec(
                name="product_a",
                dataset_ids=["unused-a"],
                variables=["a"],
                feature_group="optics",
                preprocess={"positive_quantile": None, "log": False, "add_cloud_land_masks": False},
            ),
            ProductSpec(
                name="product_b",
                dataset_ids=["unused-b"],
                variables=["b"],
                feature_group="optics",
                preprocess={"positive_quantile": None, "log": False, "add_cloud_land_masks": False},
            ),
        ],
    )

    artifacts = preprocess_matchups(config, paths.root)
    group = xr.load_dataarray(artifacts["optics"])

    assert group.sizes["variable"] == 2
    assert group["time"].values.tolist() == [0, 1]


def test_preprocess_keeps_common_ids_within_feature_group(tmp_path):
    paths = RunPaths(tmp_path / "run").ensure()
    targets = pd.DataFrame(
        {
            "Id": [1, 2],
            "lat": [40.0, 41.0],
            "lon": [1.0, 2.0],
            "time": [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-16")],
            "target": [1.0, 2.0],
        }
    )
    targets.to_csv(paths.processed / "targets.csv", index=False)

    def write_matchup(product_name: str, variable_name: str, ids: list[int]) -> None:
        values = np.ones((len(ids), 2, 2, 2), dtype=np.float32)
        dates = pd.date_range("2020-01-01", periods=2)
        ds = xr.Dataset(
            {variable_name: (("Id", "lat", "lon", "time"), values)},
            coords={"Id": ids, "lat": [0, 1], "lon": [0, 1], "time": [0, 1]},
        )
        ds = ds.assign_coords(
            {
                "lat": xr.DataArray(np.tile([40.0, 40.01], (len(ids), 1)), dims=["Id", "lat"]),
                "lon": xr.DataArray(np.tile([1.0, 1.01], (len(ids), 1)), dims=["Id", "lon"]),
                "time": xr.DataArray(np.tile(dates.values, (len(ids), 1)), dims=["Id", "time"]),
            }
        )
        ds.to_netcdf(paths.matchups / f"{product_name}.nc")

    write_matchup("product_a", "a", [1, 2])
    write_matchup("product_b", "b", [2])

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[
            ProductSpec(
                name="product_a",
                dataset_ids=["unused-a"],
                variables=["a"],
                feature_group="optics",
                preprocess={"positive_quantile": None, "log": False, "add_cloud_land_masks": False},
            ),
            ProductSpec(
                name="product_b",
                dataset_ids=["unused-b"],
                variables=["b"],
                feature_group="optics",
                preprocess={"positive_quantile": None, "log": False, "add_cloud_land_masks": False},
            ),
        ],
    )

    artifacts = preprocess_matchups(config, paths.root)
    group = xr.load_dataarray(artifacts["optics"])

    assert group["Id"].values.tolist() == [2]
    assert group.sizes["variable"] == 2


def test_min_valid_ratio_uses_selected_time_window(tmp_path):
    paths = RunPaths(tmp_path / "run").ensure()
    targets = pd.DataFrame(
        {
            "Id": [1, 2],
            "lat": [40.0, 41.0],
            "lon": [1.0, 2.0],
            "time": [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-16")],
            "target": [1.0, 2.0],
        }
    )
    targets.to_csv(paths.processed / "targets.csv", index=False)

    values = np.ones((2, 1, 1, 4), dtype=np.float32)
    values[0, :, :, :2] = np.nan
    xr.Dataset(
        {"rrs": (("Id", "lat", "lon", "time"), values)},
        coords={"Id": [1, 2], "lat": [0], "lon": [0], "time": [0, 1, 2, 3]},
    ).to_netcdf(paths.matchups / "reflectance.nc")

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        preprocess=PreprocessConfig(time_limit=2),
        products=[
            ProductSpec(
                name="reflectance",
                dataset_ids=["unused"],
                variables=["rrs"],
                feature_group="optics",
                preprocess={
                    "positive_quantile": None,
                    "log": False,
                    "add_cloud_land_masks": True,
                    "mask_kinds": ["cloud_mask", "land_mask"],
                    "min_valid_ratio": 0.3,
                    "fillna": 0.0,
                },
            )
        ],
    )

    artifacts = preprocess_matchups(config, paths.root)
    group = xr.load_dataarray(artifacts["optics"])

    assert group["Id"].values.tolist() == [2]


def test_preprocess_adds_derived_nutrient_variables_before_log1p(tmp_path):
    paths = RunPaths(tmp_path / "run").ensure()
    targets = pd.DataFrame(
        {
            "Id": [1],
            "lat": [40.0],
            "lon": [1.0],
            "time": [pd.Timestamp("2020-01-15")],
            "target": [1.0],
        }
    )
    targets.to_csv(paths.processed / "targets.csv", index=False)
    shape = (1, 1, 1, 1)
    xr.Dataset(
        {
            "nh4": (("Id", "lat", "lon", "time"), np.full(shape, 1.0, dtype=np.float32)),
            "no3": (("Id", "lat", "lon", "time"), np.full(shape, 4.0, dtype=np.float32)),
            "po4": (("Id", "lat", "lon", "time"), np.full(shape, 2.0, dtype=np.float32)),
        },
        coords={"Id": [1], "lat": [0], "lon": [0], "time": [0]},
    ).to_netcdf(paths.matchups / "nutrients.nc")
    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[
            ProductSpec(
                name="nutrients",
                dataset_ids=["unused"],
                variables=["nh4", "no3", "po4"],
                feature_group="nut",
                preprocess={
                    "positive_quantile": None,
                    "derived_variables": [
                        {"name": "din", "expression": "no3 + nh4"},
                        {"name": "n_div_p", "expression": "no3 / po4"},
                        {"name": "din_div_p", "expression": "(no3 + nh4) / po4"},
                        {"name": "nh4_div_no3", "expression": "nh4 / no3"},
                    ],
                    "log": False,
                    "log1p": True,
                    "add_cloud_land_masks": False,
                    "fillna": 0.0,
                },
            )
        ],
    )

    artifacts = preprocess_matchups(config, paths.root)
    group = xr.load_dataarray(artifacts["nut"])

    assert group["variable"].values.tolist() == [
        "nh4",
        "no3",
        "po4",
        "din",
        "n_div_p",
        "din_div_p",
        "nh4_div_no3",
    ]
    values = {
        str(name): float(group.sel(variable=name).values.reshape(-1)[0])
        for name in group["variable"].values
    }
    assert np.isclose(values["din"], np.log1p(5.0))
    assert np.isclose(values["n_div_p"], np.log1p(2.0))
    assert np.isclose(values["din_div_p"], np.log1p(2.5))
    assert np.isclose(values["nh4_div_no3"], np.log1p(0.25))


def test_preprocess_can_exclude_variables_from_log1p(tmp_path):
    paths = RunPaths(tmp_path / "run").ensure()
    targets = pd.DataFrame(
        {
            "Id": [1],
            "lat": [40.0],
            "lon": [1.0],
            "time": [pd.Timestamp("2020-01-15")],
            "target": [1.0],
        }
    )
    targets.to_csv(paths.processed / "targets.csv", index=False)
    shape = (1, 1, 1, 1)
    xr.Dataset(
        {
            "dissic": (("Id", "lat", "lon", "time"), np.full(shape, 3.0, dtype=np.float32)),
            "ph": (("Id", "lat", "lon", "time"), np.full(shape, 8.1, dtype=np.float32)),
        },
        coords={"Id": [1], "lat": [0], "lon": [0], "time": [0]},
    ).to_netcdf(paths.matchups / "carbon.nc")

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[
            ProductSpec(
                name="carbon",
                dataset_ids=["unused"],
                variables=["dissic", "ph"],
                feature_group="car",
                preprocess={
                    "positive_quantile": None,
                    "log": False,
                    "log1p": True,
                    "exclude_from_log1p": ["ph"],
                    "add_cloud_land_masks": False,
                    "fillna": 0.0,
                },
            )
        ],
    )

    artifacts = preprocess_matchups(config, paths.root)
    group = xr.load_dataarray(artifacts["car"])

    assert np.isclose(float(group.sel(variable="dissic").values.reshape(-1)[0]), np.log1p(3.0))
    assert np.isclose(float(group.sel(variable="ph").values.reshape(-1)[0]), 8.1)


def test_cnn3d_loader_preserves_cube_shape(tmp_path):
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
        ModelStageConfig(family="cnn3d", feature_groups=["optics"]),
    )
    tree_x, _ = _load_matrix_for_groups(
        datasets_dir,
        [1, 2],
        ["optics"],
        ModelStageConfig(family="random_forest", feature_groups=["optics"]),
    )

    assert cnn_x.shape == (2, 2, 2, 2, 3)
    assert np.isfinite(cnn_x).all()
    assert tree_x.shape == (2, 24)
    assert np.isnan(tree_x).sum() == 1


def test_balanced_weights_and_augmentation_preserve_targets():
    stage = ModelStageConfig(
        family="cnn3d",
        feature_groups=["optics"],
        sample_weight={"mode": "balanced", "class_boost": [1.0, 1.0, 2.0]},
        augmentation={
            "enabled": True,
            "repetitions": 2,
            "seed": 42,
            "noise_std": {"optics": [0.1, 0.0]},
        },
    )
    labels = np.array([0, 0, 1, 2])
    y = np.eye(3)[labels]
    x = np.zeros((4, 2, 2, 2, 2), dtype=np.float32)

    weights, weight_info = _make_sample_weights("classification", stage, labels)
    x_fit, y_fit, labels_fit, weights_fit, augmentation_info = _augment_training_data(
        x,
        y,
        labels,
        weights,
        stage,
        random_state=42,
    )

    assert weight_info["class_distribution"] == {"0": 2, "1": 1, "2": 1}
    assert weights[labels == 2][0] > weights[labels == 1][0]
    assert x_fit.shape[0] == 12
    assert y_fit.shape == (12, 3)
    assert labels_fit[:4].tolist() == labels.tolist()
    assert labels_fit[4:].tolist() == np.repeat(labels, 2).tolist()
    assert weights_fit.shape == (12,)
    assert augmentation_info["augmented_samples"] == 8
    assert np.allclose(y_fit.sum(axis=1), 1.0)


def test_augmentation_noise_std_mismatch_is_reported_not_fatal():
    stage = ModelStageConfig(
        family="cnn3d",
        feature_groups=["nut", "car", "phy"],
        augmentation={
            "enabled": True,
            "repetitions": 1,
            "seed": 42,
            "noise_std": {
                "nut": [0.1, 0.1],
                "car": [0.1, 0.1],
                "phy": [0.1, 0.1],
            },
        },
    )
    x = np.zeros((2, 2, 2, 2, 4), dtype=np.float32)
    y = np.array([0, 1])

    x_fit, y_fit, labels_fit, weights_fit, augmentation_info = _augment_training_data(
        x,
        y,
        labels=None,
        sample_weight=None,
        stage=stage,
        random_state=42,
    )

    assert x_fit.shape[0] == 4
    assert y_fit.tolist() == [0, 1, 0, 1]
    assert labels_fit is None
    assert weights_fit is None
    assert augmentation_info["noise_std_info"]["requested_channels"] == 6
    assert augmentation_info["noise_std_info"]["channels"] == 4
    assert augmentation_info["noise_std_info"]["adjustment"] == "truncated_to_feature_channels"


def test_multi_base_stacking_saves_stage_metrics(tmp_path):
    run_root = tmp_path / "run"
    datasets_dir = run_root / "datasets"
    datasets_dir.mkdir(parents=True)
    ids = np.arange(12)
    target_values = ids.astype(float)

    xr.DataArray(
        target_values.reshape(-1, 1),
        dims=("Id", "variable"),
        coords={"Id": ids, "variable": ["target"]},
    ).to_netcdf(datasets_dir / "target.nc")

    for group, offset in [("optics", 0.0), ("phy", 1.0), ("meta", 2.0)]:
        values = np.column_stack([target_values + offset, target_values * 0.5 + offset])
        xr.DataArray(
            values,
            dims=("Id", "variable"),
            coords={"Id": ids, "variable": [f"{group}_a", f"{group}_b"]},
        ).to_netcdf(datasets_dir / f"{group}.nc")

    config = PipelineConfig(
        target=TargetConfig(path=str(tmp_path / "unused.csv"), target_column="target"),
        products=[],
        problem=ProblemConfig(type="regression", test_size=0.25, random_state=42),
        model=ModelConfig(
            strategy="stacking",
            include_base_prediction=True,
            base_models={
                "optics": ModelStageConfig(
                    family="random_forest",
                    feature_groups=["optics"],
                    params={"n_estimators": 3, "random_state": 42},
                ),
                "physics": ModelStageConfig(
                    family="random_forest",
                    feature_groups=["phy"],
                    params={"n_estimators": 3, "random_state": 43},
                ),
            },
            final_model=ModelStageConfig(
                family="random_forest",
                feature_groups=["meta"],
                params={"n_estimators": 3, "random_state": 44},
            ),
        ),
    )

    artifacts = train_model(config, run_root)
    stage_metrics = json.loads((run_root / "metrics" / "stage_metrics.json").read_text())
    predictions = pd.read_csv(run_root / "metrics" / "predictions.csv")

    assert artifacts["base_optics_metrics"] == run_root / "metrics" / "base_optics_metrics.json"
    assert artifacts["base_physics_metrics"] == run_root / "metrics" / "base_physics_metrics.json"
    assert artifacts["base_optics_signals"] == run_root / "metrics" / "base_optics_signals.npz"
    assert artifacts["base_physics_signals"] == run_root / "metrics" / "base_physics_signals.npz"
    assert [stage["name"] for stage in stage_metrics["stages"]] == ["optics", "physics", "final"]
    assert stage_metrics["stages"][-1]["input_variant"] == "all_base_oof_signals_plus_metadata"
    assert "train_oof_metrics" in stage_metrics["stages"][-1]
    assert "train_oof_metrics" in stage_metrics["stages"][0]
    assert "base_optics_signal" in predictions.columns
    assert "base_physics_signal" in predictions.columns

    (run_root / "metrics" / "final_input_selection_metrics.json").write_text("{}", encoding="utf-8")
    (run_root / "metrics" / "final_ablation_metrics.json").write_text("{}", encoding="utf-8")
    final_artifacts = train_final_model(config, run_root)
    refreshed_stage_metrics = json.loads((run_root / "metrics" / "stage_metrics.json").read_text())

    assert final_artifacts["final_metrics"] == run_root / "metrics" / "final_metrics.json"
    assert not (run_root / "metrics" / "final_input_selection_metrics.json").exists()
    assert not (run_root / "metrics" / "final_ablation_metrics.json").exists()
    assert [stage["name"] for stage in refreshed_stage_metrics["stages"]] == ["optics", "physics", "final"]
    assert refreshed_stage_metrics["stages"][-1]["input_variant"] == "all_base_oof_signals_plus_metadata"


def test_keras_cnn_estimator_exposes_classes_for_sklearn_scorers():
    class FakeModel:
        def predict(self, x, verbose=0):
            return np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

    estimator = KerasCNN3DEstimator("classification")
    estimator._set_classes_from_target(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    estimator.model = FakeModel()

    assert estimator.classes_.tolist() == [0, 1, 2]
    assert estimator.predict(np.zeros((2, 2, 2, 2, 1))).tolist() == [1, 0]
