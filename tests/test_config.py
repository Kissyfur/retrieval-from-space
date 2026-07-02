from pathlib import Path

from retrieval_from_space.config import load_config


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
