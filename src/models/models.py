import xgboost as xgb
import keras
import numpy as np
import tensorflow as tf
import random
import copy
import pickle
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.models import BaseModel
from sklearn.model_selection import KFold

import tensorflow as tf


def epsilon_mse(epsilon=0.1):
    def loss(y_true, y_pred):
        error = tf.maximum(tf.abs(y_true - y_pred) - epsilon, 0.0)
        return tf.square(error)

    return loss


class ConvolutionalModel(BaseModel):
    def __init__(self, name='cnn'):
        self.batch = 64
        super().__init__(name=name)

    def model_factory(self, dims, kernel_sizes, output_dim, dropout=0., reg_factor=0.01, loss='mse',
                      learning_rate=0.0001, seed=42, final_activation='linear', metrics=None, **kwargs):
        # Define the model
        metrics = [] if metrics is None else metrics

        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        model = keras.Sequential(name=self.name)
        model.add(keras.layers.Normalization(name='normalization'))
        model.add(keras.layers.Reshape((-1, 1)))
        for i, dim in enumerate(dims):
            model.add(keras.layers.Conv1D(filters=dim, kernel_size=kernel_sizes[i], padding='same', activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(reg_factor)))
            model.add(keras.layers.BatchNormalization())
            if dropout != 0 and i > 0:
                model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(output_dim, kernel_initializer="glorot_uniform",
                                     activation=final_activation, kernel_regularizer=keras.regularizers.l2(reg_factor)))

        model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
        return model

    def hyperparameter_search(self, hyperparams_space, x, y, inner_splits=3, r=42, patience=5, score=None,
                              lower_is_better=True, sample_weight=None, **kwargs):
        scores_mean = []
        epochs_median = []
        if sample_weight is not None:
            sample_weight_c = sample_weight.copy()
        for mod_conf in hyperparams_space:
            scores = []
            epochs = []
            inner_loop = KFold(n_splits=inner_splits, shuffle=True, random_state=r).split(x)
            for id_train, id_val in inner_loop:
                x_train, y_train = x[id_train, :].copy(), y[id_train].copy()
                x_val, y_val = x[id_val, :].copy(), y[id_val].copy()
                if sample_weight is not None:
                    sample_weight = sample_weight_c[id_train]
                self.build_model(**mod_conf)
                hist = self.fit(x_train, y_train, x_val=x_val, y_val=y_val, patience=patience, **mod_conf, **kwargs)
                scores.append(score(y_val, self.predict(x_val)))
                epochs.append(len(hist.history['loss']))
                tf.keras.backend.clear_session()
                gc.collect()
            scores_mean.append(np.mean(scores))
            epochs_median.append(int(np.median(epochs)))
        print("scores mean for hp configurations: ", scores_mean)
        print("epochs median for hp configurations: ", epochs_median)
        best_indx = np.argmin(scores_mean) if lower_is_better else np.argmax(scores_mean)
        best_hp = copy.deepcopy(hyperparams_space[best_indx])
        best_epochs = epochs_median[best_indx]
        if "epochs" not in best_hp.keys():
            best_hp.update({"epochs": best_epochs})
        return best_hp, scores_mean[best_indx]

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save_model(self, p):
        p = p.with_suffix('.h5')
        for layer in self.model.layers:
            layer.trainable = True
        self.model.compile()
        self.model.save(p)

    def load_model(self, p):
        p = p.with_suffix('.h5')
        self.model = keras.models.load_model(p)


class Convolutional3DModel(ConvolutionalModel):
    def __init__(self, name='cnn3d'):
        super().__init__(name=name)
        self.batch = 64

    def model_factory(self, dims, kernel_sizes, output_dim, dropout=0., reg_factor=0.01, loss='mse',
                      learning_rate=0.0001, seed=42, final_activation='linear', metrics=None, **kwargs):
        metrics = [] if metrics is None else metrics
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        model = keras.Sequential(name=self.name)
        # model.add(keras.layers.Normalization(name='normalization'))
        for i, dim in enumerate(dims):
            model.add(keras.layers.Conv3D(filters=dim, kernel_size=kernel_sizes[i], padding='same', activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(reg_factor)
                                          ))
            model.add(keras.layers.BatchNormalization())
            if dropout != 0 and i > 0:
                model.add(keras.layers.Dropout(dropout))

        # Global pooling and output layer
        model.add(keras.layers.GlobalAveragePooling3D())
        model.add(keras.layers.Dense(output_dim, kernel_initializer="glorot_uniform", activation=final_activation,
                                     kernel_regularizer=keras.regularizers.l2(reg_factor)
                                     ))
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
        return model

    def fit(self, x, y, x_val=None, y_val=None, epochs=5000, cb=None, patience=5, verbose=0, sample_weight=None,
            **kwargs):
        if cb is None:
            cb = []
        val_data = None
        if x_val is not None:
            val_data = (x_val, y_val)
            if patience != 0:
                cb += [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
        h = self.model.fit(x, y, validation_data=val_data, epochs=epochs, shuffle=True, verbose=verbose,
                           batch_size=self.batch, callbacks=cb, sample_weight=sample_weight)
        return h


class Convolutional3DModelClassification(Convolutional3DModel):
    def __init__(self, name='cnn3d'):
        super().__init__(name=name)
        self.batch = 64

    def model_factory(self, dims, kernel_sizes, output_dim, dropout=0., reg_factor=0.01,
                      loss='categorical_crossentropy',
                      learning_rate=0.0001, seed=42, final_activation='softmax', **kwargs):
        model = super().model_factory(dims, kernel_sizes, output_dim, dropout, reg_factor=reg_factor, loss=loss,
                                      learning_rate=learning_rate, seed=seed, final_activation=final_activation,
                                      **kwargs)
        return model


class RandomForestModel(BaseModel):
    def __init__(self, name='rf'):
        super().__init__(name=name)

    def model_factory(self, **kwargs):
        model = RandomForestRegressor(**kwargs)
        return model

    def save_model(self, p):
        p = p.with_suffix('.pkl')
        with open(p, "wb") as f:
            pickle.dump(self.model, f)

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y)

    def load_model(self, p):
        p = p.with_suffix('.pkl')
        with open(p, "rb") as f:
            self.model = pickle.load(f)

    def hyperparameter_search(self, hyperparams_space, x, y, n_iter=50, random_state=42,
                              inner_splits=3, repetitions=100, **kwargs):
        model = self.model_factory(random_state=random_state)
        # x, y = augment_data(x, y, replicate=repetitions)

        randomized_search = RandomizedSearchCV(
            estimator=model, param_distributions=hyperparams_space, n_iter=n_iter,
            cv=inner_splits, random_state=random_state, n_jobs=-1)
        randomized_search.fit(x, y)
        return randomized_search.best_params_, randomized_search.best_score_


class XGBModel(RandomForestModel):
    def __init__(self, name='xgb'):
        super().__init__(name=name)

    def fit(self, x, y, eval_set=None, **kwargs):
        return self.model.fit(x, y, eval_set=eval_set)

    def model_factory(self, **kwargs):
        model = xgb.XGBRegressor(**kwargs)
        return model

    def save_model(self, p):
        p = p.with_suffix('.json')
        self.model.save_model(p)

    def load_model(self, p):
        p = p.with_suffix('.json')
        self.model = xgb.XGBRegressor()
        self.model.load_model(p)


class XGBModelClassifier(XGBModel):
    def __init__(self, name='xgbClassi'):
        super().__init__(name=name)

    def model_factory(self, **kwargs):
        model = xgb.XGBClassifier(**kwargs)
        return model
