import logging

from pathlib import Path

logging.basicConfig(level=logging.INFO)


class BaseModel:

    def __init__(self, name='baseModel'):
        self.name = name
        self.model = None

    def predict(self, x):
        py = self.model.predict(x)
        return py

    def build_model(self, **kwargs):
        if kwargs is None:
            return
        self.model = self.model_factory(**kwargs)

    def model_factory(self, **kwargs):
        return

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)

    def save_model(self, p):
        return

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        p = p / self.name
        self.save_model(p)

    def load_model(self, p):
        return

    def load(self, path):
        p = Path(path)
        p = p / self.name
        self.load_model(p)

    def hyperparameter_search(self, hyperparams_space, x, y, minimizing_function, **kwargs):
        return None
