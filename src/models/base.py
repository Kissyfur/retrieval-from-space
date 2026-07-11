from __future__ import annotations

from pathlib import Path


class BaseModel:
    def __init__(self, name: str = "base_model"):
        self.name = name
        self.model = None

    def build_model(self, **kwargs):
        self.model = self.model_factory(**kwargs)
        return self.model

    def model_factory(self, **kwargs):
        raise NotImplementedError

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path: str | Path):
        raise NotImplementedError

    def load(self, path: str | Path):
        raise NotImplementedError
