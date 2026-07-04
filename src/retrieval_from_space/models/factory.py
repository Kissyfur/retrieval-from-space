from __future__ import annotations

from retrieval_from_space.config import ModelStageConfig
from retrieval_from_space.models.cnn import KerasCNN3DEstimator
from retrieval_from_space.models.tree import random_forest, xgboost_model


def create_model(problem_type: str, config: ModelStageConfig, params: dict | None = None):
    candidate_params = {} if params is None else dict(params)
    family = str(
        candidate_params.pop("family", candidate_params.pop("model_family", config.family))
    ).lower()
    model_params = {**config.params, **candidate_params}
    if family in {"random_forest", "rf", "auto"}:
        return random_forest(problem_type, **model_params)
    if family in {"xgboost", "xgb"}:
        return xgboost_model(problem_type, **model_params)
    if family in {"cnn3d", "3d_cnn"}:
        return KerasCNN3DEstimator(problem_type, **model_params)
    raise ValueError(f"Unsupported model family: {config.family}")
