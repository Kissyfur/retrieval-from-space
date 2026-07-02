from __future__ import annotations

import random

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
):
    import keras
    import tensorflow as tf

    metrics = [] if metrics is None else metrics
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    model = keras.Sequential(name="cnn3d")
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
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
    return model
