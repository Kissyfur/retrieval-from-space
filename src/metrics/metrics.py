import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)

logging.basicConfig(level=logging.INFO)


def drop_nans(ty, py):
    t_not_nans = ~np.isnan(ty)
    p_not_nans = ~np.isnan(py)
    not_nans = t_not_nans * p_not_nans
    ty = ty[not_nans]
    py = py[not_nans]
    return ty, py


def nans_inf_in_data(arr):
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return True
    return False


def exponential_values(arr, transport):
    arr = np.exp(arr) + transport
    return arr


def exponential_r2(y, py, transport=-100):
    if nans_inf_in_data(py):
        return -999
    y = exponential_values(y, transport)
    py = exponential_values(py, transport)
    return r2_score(y, py)

# return list(mean_absolute_error(y, py, multioutput='raw_values'))


def custom_r2(y, py):
    if nans_inf_in_data(py):
        return -999
    return r2_score(y, py)


def exponential_mape(y, py, transport=-100):
    if nans_inf_in_data(py):
        return 500
    y = exponential_values(y, transport)
    py = exponential_values(py, transport)
    return mean_absolute_percentage_error(y, py)


def exponential_mpe(y, py, transport=-100):
    if nans_inf_in_data(py):
        return 500
    y = exponential_values(y, transport)
    py = exponential_values(py, transport)
    return np.mean((y - py)/y)


def exponential_mae(y, py, transport=-100):
    if nans_inf_in_data(py):
        print("NaN in py")
        return 500
    y = exponential_values(y, transport)
    py = exponential_values(py, transport)
    return mean_absolute_error(y, py)


def exponential_me(y, py, transport=-100):
    if nans_inf_in_data(py):
        print("NaN in py")
        return 500
    y = exponential_values(y, transport)
    py = exponential_values(py, transport)
    return np.mean(y - py)


def custom_mse(y, py):
    if nans_inf_in_data(py):
        print("NaN in py")
        return 500
    return mean_squared_error(y, py)


def mae_mse_loss(y, py):
    mean_absolute_error(y, py) + mean_squared_error(y, py)


class Metrics:
    M_FUNCTIONS = {
        "MSE": custom_mse,
        "R2": r2_score,
        "MAPE": exponential_mape,
        "MAE": exponential_mae,
        "MPE": exponential_mpe,
        "ME": exponential_me
    }
    MET_NAMES = M_FUNCTIONS.keys()

    def __init__(self, met_names=MET_NAMES):
        self.metrics = {m_name:  self.M_FUNCTIONS[m_name] for m_name in met_names}

    def compute_metrics_df(self, ty, py):
        m = self.compute_metrics(ty.values, py.values)
        return pd.DataFrame(m, index=ty.columns).T

    def compute_metrics(self, ty, py):
        cols = ty.shape[1]
        mets = []
        for col in range(cols):
            t, p = drop_nans(ty[:, col], py[:, col])
            if len(t) == 0:
                logging.info(f"Can not compute metrics on column {col} due to NaN's")
                mets.append([np.nan for met in self.metrics])
                continue
            mets.append({met_name: met(t, p) for met_name, met in self.metrics.items()})
        return mets
