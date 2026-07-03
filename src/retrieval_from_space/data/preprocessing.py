from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from retrieval_from_space.config import PipelineConfig, ProductSpec
from retrieval_from_space.data.targets import TARGET, load_target_table, metadata_to_dataarray, target_to_dataarray
from retrieval_from_space.features.masks import get_cloud_and_land_masks, valid_water_coverage
from retrieval_from_space.features.transforms import interpolate_dataset, positive_quantile

VARIABLE = "variable"
ORDERED_CUBE_DIMS = ("Id", "lat", "lon", "time", VARIABLE)


def _option(product: ProductSpec, key: str, default: Any) -> Any:
    return product.preprocess.get(key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _select_time(data: xr.DataArray, limit: int | None) -> xr.DataArray:
    if limit is None or "time" not in data.dims:
        return data
    return data.isel(time=range(min(limit, data.sizes["time"])))


def _use_relative_cube_coordinates(data: xr.DataArray) -> xr.DataArray:
    coords = {
        dim: np.arange(data.sizes[dim], dtype=np.int32)
        for dim in ("lat", "lon", "time")
        if dim in data.dims
    }
    return data.assign_coords(coords) if coords else data


def _drop_duplicate_mask_variables(data: xr.DataArray) -> xr.DataArray:
    if VARIABLE not in data.coords:
        return data
    seen_masks = set()
    keep_indices = []
    for index, name in enumerate(data[VARIABLE].values):
        name = str(name)
        if name in {"cloud_mask", "land_mask"}:
            if name in seen_masks:
                continue
            seen_masks.add(name)
        keep_indices.append(index)
    return data.isel({VARIABLE: keep_indices})


def _ordered_common_ids(arrays: list[xr.DataArray], group_name: str) -> np.ndarray:
    if not arrays or any("Id" not in array.dims for array in arrays):
        return np.array([])
    first_ids = arrays[0]["Id"].values
    other_id_sets = [set(array["Id"].values.tolist()) for array in arrays[1:]]
    common_ids = [id_ for id_ in first_ids if all(id_ in ids for ids in other_id_sets)]
    if not common_ids:
        raise ValueError(
            f"No common Id values remain after preprocessing products in feature group '{group_name}'. "
            "Check unmatched observations and product-level valid-data filters."
        )
    return np.asarray(common_ids, dtype=first_ids.dtype)


def _align_to_common_ids(arrays: list[xr.DataArray], group_name: str) -> list[xr.DataArray]:
    if len(arrays) <= 1 or any("Id" not in array.dims for array in arrays):
        return arrays
    common_ids = _ordered_common_ids(arrays, group_name)
    return [array.sel(Id=common_ids) for array in arrays]


def _as_dataarray(ds: xr.Dataset, product: ProductSpec) -> xr.DataArray:
    variables = product.variables or list(ds.data_vars)
    variables = [product.rename_variables.get(var, var) for var in variables]
    ds = ds[variables]
    if product.rename_variables:
        rename_map = {old: new for old, new in product.rename_variables.items() if old in ds.data_vars}
        ds = ds.rename(rename_map)
    return ds.to_array(dim=VARIABLE)


def _safe_log(data: xr.DataArray) -> xr.DataArray:
    return np.log(data.where(data > 0))


def _safe_log1p(data: xr.DataArray) -> xr.DataArray:
    return np.log1p(data.where(data > -1))


def _prepare_product_array(
    ds: xr.Dataset,
    product: ProductSpec,
    defaults: PipelineConfig,
) -> xr.DataArray:
    interpolate_dims = tuple(_option(product, "interpolate_dims", ()))
    if interpolate_dims:
        ds = interpolate_dataset(ds, interpolate_dims)

    data = _as_dataarray(ds, product)
    add_masks = bool(_option(product, "add_cloud_land_masks", defaults.preprocess.add_cloud_land_masks))
    cloud_mask = land_mask = None
    if add_masks:
        cloud_mask, land_mask = get_cloud_and_land_masks(data, variable_dim=VARIABLE)

    quantile = _option(product, "positive_quantile", defaults.preprocess.positive_quantile)
    if quantile is not None:
        quantile_dims = tuple(
            _option(product, "positive_quantile_dims", ("Id", "time", "lat", "lon"))
        )
        data = positive_quantile(data, quantile=float(quantile), dims=quantile_dims)

    if bool(_option(product, "log", defaults.preprocess.log_products)):
        data = _safe_log(data)
    if bool(_option(product, "log1p", False)):
        data = _safe_log1p(data)

    if bool(_option(product, "prefix_variables", defaults.preprocess.prefix_variables)):
        names = [f"{product.name}:{name}" for name in data[VARIABLE].values]
        data = data.assign_coords({VARIABLE: names})

    arrays = [data]
    if cloud_mask is not None and land_mask is not None:
        mask_kinds = {str(value) for value in _as_list(_option(product, "mask_kinds", ["cloud_mask", "land_mask"]))}
        if "cloud_mask" in mask_kinds:
            arrays.append(cloud_mask)
        if "land_mask" in mask_kinds:
            arrays.append(land_mask)
    data = xr.concat(arrays, dim=VARIABLE, coords="minimal")

    ordered_dims = tuple(dim for dim in ORDERED_CUBE_DIMS if dim in data.dims)
    data = data.transpose(*ordered_dims)

    min_valid_ratio = _option(product, "min_valid_ratio", defaults.preprocess.min_valid_ratio)
    if min_valid_ratio is not None and "cloud_mask" in data[VARIABLE].values and "land_mask" in data[VARIABLE].values:
        ratio = valid_water_coverage(data.sel({VARIABLE: "cloud_mask"}), data.sel({VARIABLE: "land_mask"}))
        data = data.isel(Id=(ratio >= float(min_valid_ratio)).values)

    time_limit = _option(product, "time_limit", defaults.preprocess.time_limit)
    data = _select_time(data, time_limit)
    data = _use_relative_cube_coordinates(data)

    fillna = _option(product, "fillna", defaults.preprocess.fillna)
    if fillna is not None:
        data = data.fillna(fillna)
    return data


def transform_target(data: xr.DataArray, transform: str, offset: float = 0.0) -> xr.DataArray:
    transform = transform.lower()
    if transform == "none":
        return data
    if transform == "log":
        shifted = data + offset
        invalid_count = int((shifted <= 0).sum().item())
        if invalid_count:
            min_value = float(data.min(skipna=True).item())
            raise ValueError(
                "Cannot apply problem.target_transform: log because the target contains "
                f"{invalid_count} values where target + offset is non-positive. "
                f"Minimum target value: {min_value}; offset: {offset}. Use a larger "
                "problem.target_transform_offset, set target_transform: none if the target "
                "is already logged, or filter/remove invalid target rows before preprocessing."
            )
        return np.log(shifted)
    if transform == "log1p":
        shifted = data + offset
        invalid_count = int((shifted <= -1).sum().item())
        if invalid_count:
            min_value = float(data.min(skipna=True).item())
            raise ValueError(
                "Cannot apply problem.target_transform: log1p because the target contains "
                f"{invalid_count} values where target + offset is less than or equal to -1. "
                f"Minimum target value: {min_value}; offset: {offset}."
            )
        return np.log1p(shifted)
    raise ValueError(f"Unsupported target transform: {transform}")


def preprocess_matchups(config: PipelineConfig, run_root: str | Path) -> dict[str, Path]:
    run_root = Path(run_root)
    datasets_dir = run_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    target_table_path = run_root / "processed" / "targets.csv"
    targets = pd.read_csv(target_table_path, parse_dates=["time"]) if target_table_path.exists() else load_target_table(config.target)

    target_da = target_to_dataarray(targets, target_name=TARGET)
    target_da = transform_target(
        target_da,
        config.problem.target_transform,
        offset=config.problem.target_transform_offset,
    )
    target_path = datasets_dir / "target.nc"
    target_da.to_netcdf(target_path)

    meta = metadata_to_dataarray(
        targets,
        config.target.metadata_columns,
        include_spatial_metadata=config.target.include_spatial_metadata,
        include_day_metadata=config.target.include_day_metadata,
        include_cyclic_day_metadata=config.target.include_cyclic_day_metadata,
    )
    meta_path = datasets_dir / "meta.nc"
    meta.to_netcdf(meta_path)

    grouped: dict[str, list[xr.DataArray]] = defaultdict(list)
    artifacts: dict[str, Path] = {"target": target_path, "meta": meta_path}

    for product in config.products:
        matchup_path = run_root / "processed" / "matchups" / f"{product.name}.nc"
        if not matchup_path.exists():
            continue
        ds = xr.load_dataset(matchup_path)
        group_name = product.feature_group or product.preprocess.get("feature_group") or product.name
        grouped[group_name].append(_prepare_product_array(ds, product, config))

    for group_name, arrays in grouped.items():
        arrays = _align_to_common_ids(arrays, group_name)
        if len(arrays) == 1:
            group = arrays[0]
        else:
            group = xr.concat(arrays, dim=VARIABLE, coords="minimal", compat="override", join="override")
        group = _drop_duplicate_mask_variables(group)
        path = datasets_dir / f"{group_name}.nc"
        group.to_netcdf(path)
        artifacts[group_name] = path
    return artifacts
