import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from retrieval_from_space.config import ModelStageConfig, PipelineConfig, ProductSpec, TargetConfig, load_config
from retrieval_from_space.models.training import interval_soft_labeling
from retrieval_from_space.models.training import _load_matrix_for_groups
from retrieval_from_space.paths import RunPaths
from retrieval_from_space.pipeline.download import download_products
from retrieval_from_space.pipeline.matchup import create_matchups
from retrieval_from_space.state import PipelineState
from retrieval_from_space.data.preprocessing import preprocess_matchups, transform_target
from retrieval_from_space.features.transforms import positive_quantile


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
    assert config.matchup.time_window_days == 14
    assert config.model.family == "cnn3d"
    assert config.model.feature_groups == ["optics"]
    assert len(config.model.hyperparameter_search.candidates) == 3
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
        "retrieval_from_space.pipeline.matchup.load_target_table",
        lambda target: observations,
    )

    def fake_create_product_matchups(product, targets, matchup, raw_path=None):
        captured["raw_path"] = raw_path
        return None, targets[["Id"]].copy()

    monkeypatch.setattr(
        "retrieval_from_space.pipeline.matchup.create_product_matchups",
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
    assert tree_x.shape == (2, 24)
