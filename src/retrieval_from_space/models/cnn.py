from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def epsilon_mse(epsilon: float = 0.1):
    import tensorflow as tf

    def loss(y_true, y_pred):
        error = tf.maximum(tf.abs(y_true - y_pred) - epsilon, 0.0)
        return tf.square(error)

    return loss


def build_cnn3d(
    dims,
    kernel_sizes,
    output_dim,
    dropout: float = 0.0,
    reg_factor: float = 0.01,
    loss: str = "mse",
    learning_rate: float = 0.0001,
    seed: int = 42,
    final_activation: str = "linear",
    metrics=None,
    weighted_metrics=None,
    name: str = "cnn3d",
    **kwargs,
):
    import keras
    import tensorflow as tf

    metrics = [] if metrics is None else metrics
    weighted_metrics = [] if weighted_metrics is None else weighted_metrics
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    model = keras.Sequential(name=name)
    for index, dim in enumerate(dims):
        model.add(
            keras.layers.Conv3D(
                filters=dim,
                kernel_size=kernel_sizes[index],
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(reg_factor),
            )
        )
        model.add(keras.layers.BatchNormalization())
        if dropout and index > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.GlobalAveragePooling3D())
    model.add(
        keras.layers.Dense(
            output_dim,
            kernel_initializer="glorot_uniform",
            activation=final_activation,
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


def _format_log_value(value):
    try:
        return f"{float(np.asarray(value).reshape(-1)[0]):.4g}"
    except (TypeError, ValueError, IndexError):
        return value


class KerasCNN3DEstimator:
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
            "dims",
            "kernel_sizes",
            "output_dim",
            "dropout",
            "reg_factor",
            "loss",
            "learning_rate",
            "seed",
            "final_activation",
            "metrics",
            "weighted_metrics",
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
        fit_params.setdefault("epochs", 5000)
        fit_params.setdefault("batch_size", 64)
        fit_params.setdefault("patience", 3)
        fit_params.setdefault("validation_split", 0.15)
        fit_params.setdefault("verbose", 0)
        return build_params, fit_params

    def fit(self, x, y, sample_weight=None):
        import keras
        from tqdm.auto import tqdm

        build_params, fit_params = self._split_params()
        self._set_classes_from_target(y)
        self.model = build_cnn3d(**build_params)
        callbacks = []
        patience = int(fit_params.pop("patience"))
        show_progress = bool(fit_params.pop("show_progress", True))
        progress_description = str(fit_params.pop("progress_description", self.params.get("name", "cnn3d")))
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
            raise ValueError("The CNN model has not been fitted.")
        prediction = np.asarray(self.model.predict(x, verbose=0))
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
            raise ValueError("The CNN model has not been fitted.")
        path = Path(path).with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return path


def _make_tqdm_epoch_progress(keras, tqdm_factory, epochs: int, description: str, leave: bool):
    class TqdmEpochProgress(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._bar = None

        def on_train_begin(self, logs=None):
            self._bar = tqdm_factory(
                total=epochs,
                desc=description,
                unit="epoch",
                leave=leave,
            )

        def on_epoch_end(self, epoch, logs=None):
            if self._bar is None:
                return
            logs = logs or {}
            keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
            postfix = {key: _format_log_value(logs[key]) for key in keys if key in logs}
            if postfix:
                self._bar.set_postfix(postfix)
            self._bar.update(1)

        def on_train_end(self, logs=None):
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    return TqdmEpochProgress()
