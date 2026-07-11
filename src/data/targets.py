from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from src.config import TargetConfig
from src.features.coordinates import dms_to_decimal
from src.features.temporal import day_to_circle_x, day_to_circle_y

ID = "Id"
LAT = "lat"
LON = "lon"
TIME = "time"
TARGET = "target"


def load_target_table(config: TargetConfig) -> pd.DataFrame:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Target table does not exist: {path}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        data = pd.read_excel(path, sheet_name=config.sheet_name)
    else:
        data = pd.read_csv(path)

    rename_map = {
        config.id_column: ID,
        config.lat_column: LAT,
        config.lon_column: LON,
        config.time_column: TIME,
        config.target_column: TARGET,
    }
    data = data.rename(columns=rename_map)
    if ID not in data.columns:
        data[ID] = range(len(data))

    required = {ID, LAT, LON, TIME, TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Target table is missing required columns: {sorted(missing)}")

    data[LAT] = data[LAT].map(dms_to_decimal).astype(float)
    data[LON] = data[LON].map(dms_to_decimal).astype(float)
    data[TIME] = pd.to_datetime(data[TIME])
    data = data.dropna(subset=[ID, LAT, LON, TIME, TARGET]).copy()
    return data


def save_standard_target_table(data: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def target_to_dataarray(data: pd.DataFrame, target_name: str = TARGET) -> xr.DataArray:
    target = data[[ID, TARGET]].set_index(ID)
    target = target.rename(columns={TARGET: target_name})
    return target.to_xarray().to_dataarray(dim="variable").transpose(ID, "variable")


def metadata_to_dataarray(
    data: pd.DataFrame,
    metadata_columns: list[str] | None = None,
    include_spatial_metadata: bool = True,
    include_day_metadata: bool = True,
    include_cyclic_day_metadata: bool = True,
) -> xr.DataArray:
    metadata_columns = [] if metadata_columns is None else metadata_columns
    missing_metadata = [col for col in metadata_columns if col not in data.columns]
    if missing_metadata:
        raise ValueError(f"Target table is missing configured metadata columns: {missing_metadata}")
    columns = [TIME] + list(metadata_columns)
    if include_spatial_metadata:
        columns = [LAT, LON] + columns
    meta = data[[ID] + columns].copy()
    meta[TIME] = pd.to_datetime(meta[TIME])
    meta["day"] = meta[TIME].dt.dayofyear
    meta["x_day"] = meta["day"].map(day_to_circle_x)
    meta["y_day"] = meta["day"].map(day_to_circle_y)
    base_columns = []
    if include_spatial_metadata:
        base_columns.extend([LAT, LON])
    if include_cyclic_day_metadata:
        base_columns.extend(["x_day", "y_day"])
    if include_day_metadata:
        base_columns.append("day")
    numeric_columns = [
        col
        for col in base_columns + metadata_columns
        if col in meta.columns and col != TIME
    ]
    return meta[[ID] + numeric_columns].set_index(ID).to_xarray().to_dataarray(
        dim="variable"
    ).transpose(ID, "variable")
