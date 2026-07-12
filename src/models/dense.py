from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.models.cnn import _make_tqdm_epoch_progress


def build_dense(
    input_dim: int,
    output_dim: int = 1,
    hidden_units=None,
    dropout: float = 0.0,
    reg_factor: float = 0.0,
    loss: str = "mse",
    learning_rate: float = 0.001,
    seed: int = 42,
    final_activation: str = "linear",
    metrics=None,
    weighted_metrics=None,
    batch_norm: bool = False,
    name: str = "dense",
    **kwargs,
):
    import keras
    import tensorflow as tf

    hidden_units = [32, 16] if hidden_units is None else list(hidden_units)
    metrics = [] if metrics is None else metrics
    weighted_metrics = [] if weighted_metrics is None else weighted_metrics
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    model = keras.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(int(input_dim),)))
    for units in hidden_units:
        model.add(
            keras.layers.Dense(
                int(units),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(reg_factor),
            )
        )
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        if dropout:
            model.add(keras.layers.Dropout(float(dropout)))
    model.add(
        keras.layers.Dense(
            int(output_dim),
            activation=final_activation,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=keras.regularizers.l2(reg_factor),
        )
    )
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    return model


class KerasDenseEstimator:
    def __init__(self, problem_type: str, **params):
        self.problem_type = problem_type
        self.params = dict(params)
        self.model = None
        self.history = None
        self._estimator_type = "classifier" if problem_type == "classification" else "regressor"
        if self.problem_type == "classification":
            self.classes_ = None

    def _split_params(self) -> tuple[dict, dict]:
        build_keys = {
            "input_dim",
            "output_dim",
            "hidden_units",
            "dropout",
            "reg_factor",
            "loss",
            "learning_rate",
            "seed",
            "final_activation",
            "metrics",
            "weighted_metrics",
            "batch_norm",
            "name",
        }
        fit_keys = {
            "epochs",
            "batch_size",
            "patience",
            "validation_split",
            "verbose",
            "show_progress",
            "progress_description",
            "progress_leave",
        }
        build_params = {key: value for key, value in self.params.items() if key in build_keys}
        fit_params = {key: value for key, value in self.params.items() if key in fit_keys}
        fit_params.setdefault("epochs", 500)
        fit_params.setdefault("batch_size", 64)
        fit_params.setdefault("patience", 20)
        fit_params.setdefault("validation_split", 0.15)
        fit_params.setdefault("verbose", 0)
        return build_params, fit_params

    def _prepare_x(self, x):
        nan_fill = float(self.params.get("nan_fill", 0.0))
        return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=nan_fill)

    def fit(self, x, y, sample_weight=None):
        import keras
        from tqdm.auto import tqdm

        x = self._prepare_x(x)
        build_params, fit_params = self._split_params()
        self._set_classes_from_target(y)
        build_params.setdefault("input_dim", x.shape[1])
        if self.problem_type == "classification":
            build_params.setdefault("output_dim", len(self.classes_))
            build_params.setdefault("loss", "categorical_crossentropy")
            build_params.setdefault("final_activation", "softmax")
        else:
            build_params.setdefault("output_dim", 1)
            build_params.setdefault("loss", "mse")
            build_params.setdefault("final_activation", "linear")
        self.model = build_dense(**build_params)

        callbacks = []
        patience = int(fit_params.pop("patience"))
        show_progress = bool(fit_params.pop("show_progress", True))
        progress_description = str(fit_params.pop("progress_description", self.params.get("name", "dense")))
        progress_leave = bool(fit_params.pop("progress_leave", False))
        epochs = int(fit_params.get("epochs", 0))
        verbose = int(fit_params.get("verbose", 0))
        validation_split = float(fit_params.get("validation_split", 0.0))
        if patience > 0 and validation_split > 0:
            callbacks.append(keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True))
        if show_progress and verbose == 0 and epochs > 0:
            callbacks.append(
                _make_tqdm_epoch_progress(
                    keras,
                    tqdm,
                    epochs=epochs,
                    description=progress_description,
                    leave=progress_leave,
                )
            )
        self.history = self.model.fit(
            x,
            y,
            shuffle=True,
            callbacks=callbacks,
            sample_weight=sample_weight,
            **fit_params,
        )
        return self

    def _set_classes_from_target(self, y) -> None:
        if self.problem_type != "classification":
            return
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        else:
            self.classes_ = np.unique(y.reshape(-1))

    def predict_proba(self, x):
        if self.model is None:
            raise ValueError("The dense model has not been fitted.")
        prediction = np.asarray(self.model.predict(self._prepare_x(x), verbose=0))
        if self.problem_type == "classification":
            return prediction
        return prediction.reshape(-1, 1)

    def predict(self, x):
        prediction = self.predict_proba(x)
        if self.problem_type == "classification":
            labels = np.argmax(prediction, axis=1)
            classes = getattr(self, "classes_", None)
            if classes is not None:
                return np.asarray(classes)[labels]
            return labels
        return prediction.reshape(-1)

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise ValueError("The dense model has not been fitted.")
        path = Path(path).with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return path
