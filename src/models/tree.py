from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest(problem_type: str, **params):
    params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1, **params}
    if problem_type == "classification":
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)


def save_pickle_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def load_pickle_model(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
