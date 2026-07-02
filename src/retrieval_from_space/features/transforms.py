from __future__ import annotations

import itertools

import numpy as np
import xarray as xr


def positive_quantile(
    data: xr.DataArray,
    quantile: float = 0.01,
    dims: tuple[str, ...] = ("Id", "time", "lat", "lon"),
) -> xr.DataArray:
    dims = tuple(dim for dim in dims if dim in data.dims)
    quant = data.where(data > 0).quantile(quantile, dim=dims)
    return np.maximum(data, quant)


def interpolate_dataset(ds: xr.Dataset, dims: tuple[str, ...] = ("lat", "lon")) -> xr.Dataset:
    interpolated = ds.copy()
    for var_name in interpolated.data_vars:
        for dim in dims:
            if dim in interpolated[var_name].dims:
                interpolated[var_name] = interpolated[var_name].interpolate_na(
                    dim=dim, method="linear", use_coordinate=False
                )
    return interpolated


def monthly_anomaly(data: xr.DataArray) -> xr.DataArray:
    """Subtract a global monthly mean while preserving 2D time coordinates."""
    if "time" not in data.coords:
        return data
    months_map = data.time.dt.month
    climatology = [
        data.where(months_map == month).mean(skipna=True) for month in range(1, 13)
    ]
    climatology_da = xr.DataArray(climatology, coords={"month": range(1, 13)}, dims="month")
    return data - climatology_da.sel(month=months_map)


def replicate_data(data: xr.DataArray, dim: str = "Id", replicate: int = 10) -> xr.DataArray:
    return xr.concat([data] * replicate, dim=dim)


def apply_noise(
    data: xr.DataArray,
    std: float = 0.1,
    std_dims: tuple[str, ...] | None = None,
    seed: int = 42,
    add: bool = True,
) -> xr.DataArray:
    if std_dims is None:
        return data
    rng = np.random.default_rng(seed)
    sizes = tuple(data.sizes[dim] for dim in std_dims)
    center = 0.0 if add else 1.0
    noise = xr.DataArray(rng.normal(center, std, size=sizes), dims=std_dims)
    return data + noise if add else data * noise


def augment_data(
    data: xr.DataArray,
    repetitions: int,
    std: float,
    std_dims: tuple[str, ...],
    seed: int = 42,
) -> xr.DataArray:
    augmented = replicate_data(data, dim="Id", replicate=repetitions)
    augmented = apply_noise(augmented, std=std, std_dims=std_dims, seed=seed, add=True)
    return xr.concat([data, augmented], dim="Id")


def random_paired_combinations(
    data: xr.DataArray,
    target: xr.DataArray,
    dim: str = "Id",
    factor: float = 0.9,
    seed: int = 42,
) -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(seed)
    n = len(data[dim].values)
    mixing_factor = xr.DataArray(rng.uniform(factor, 1.0, n), dims=dim)
    mixing_indices = rng.choice(n, n)
    mixed_data = mixing_factor * data + ((1 - mixing_factor) * data.isel({dim: mixing_indices})).values
    mixed_target = mixing_factor * target + ((1 - mixing_factor) * target.isel({dim: mixing_indices})).values
    return mixed_data, mixed_target


def fill_value_2d(arr: np.ndarray, value: float, percent: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    copied = arr.copy()
    num_changes = int(np.prod(arr.shape) * percent)
    all_coords = list(itertools.product(*[range(size) for size in arr.shape]))
    coords = np.array(all_coords)[rng.choice(len(all_coords), num_changes, replace=False)]
    copied[coords[:, 0], coords[:, 1]] = value
    return copied
