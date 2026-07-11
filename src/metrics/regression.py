from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "me": np.nan}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "me": float(np.mean(y_true - y_pred)),
    }
