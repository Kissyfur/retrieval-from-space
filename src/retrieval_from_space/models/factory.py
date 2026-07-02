from __future__ import annotations

from retrieval_from_space.config import ModelStageConfig
from retrieval_from_space.models.tree import random_forest, xgboost_model


def create_model(problem_type: str, config: ModelStageConfig, params: dict | None = None):
    family = config.family.lower()
    model_params = {**config.params, **({} if params is None else params)}
    if family in {"random_forest", "rf", "auto"}:
        return random_forest(problem_type, **model_params)
    if family in {"xgboost", "xgb"}:
        return xgboost_model(problem_type, **model_params)
    if family in {"cnn3d", "3d_cnn"}:
        raise ValueError("cnn3d is available as a builder, but automated CLI training currently supports tree models.")
    raise ValueError(f"Unsupported model family: {config.family}")
