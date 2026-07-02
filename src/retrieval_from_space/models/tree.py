from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest(problem_type: str, **params):
    params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1, **params}
    if problem_type == "classification":
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)


def xgboost_model(problem_type: str, **params):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("The xgboost model family requires xgboost.") from exc
    if problem_type == "classification":
        defaults = {"eval_metric": "mlogloss", "random_state": 42}
        return xgb.XGBClassifier(**{**defaults, **params})
    defaults = {"random_state": 42}
    return xgb.XGBRegressor(**{**defaults, **params})


def save_pickle_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def load_pickle_model(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
