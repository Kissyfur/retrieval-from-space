from __future__ import annotations

from src.config import ModelConfig
from src.models.cnn import KerasCNN3DEstimator
from src.models.dense import KerasDenseEstimator
from src.models.tree import random_forest


REPORT_PARAM_KEYS = {
    "cv_epochs",
    "cv_epochs_mean",
    "cv_epochs_median",
    "final_training_epoch_source",
}


def create_model(problem_type: str, config: ModelConfig, params: dict | None = None):
    family = config.family.lower()
    model_params = {**config.params, **({} if params is None else params)}
    model_params = {key: value for key, value in model_params.items() if key not in REPORT_PARAM_KEYS}
    if family in {"random_forest", "rf", "auto"}:
        return random_forest(problem_type, **model_params)
    if family in {"cnn3d", "3d_cnn"}:
        return KerasCNN3DEstimator(problem_type, **model_params)
    if family in {"dense", "mlp", "dense_nn", "tabular_nn"}:
        return KerasDenseEstimator(problem_type, **model_params)
    raise ValueError(f"Unsupported model family: {config.family}")
