import pandas as pd
import xarray as xr
import re
import numpy as np
import itertools as it

from tqdm.notebook import tqdm

LAT, LON, TIME = 'lat', 'lon', 'time'

# Thresholds for mathcup
TIME_TH = pd.Timedelta(days=1)
LAT_TH = 0.1
LON_TH = 0.1

# Matchup region extension (window)
TIME_WINDOW = pd.Timedelta(days=1)
LAT_WINDOW = 0.06
LON_WINDOW = 0.06


def get_cloud_and_land_masks(da, dim='variable', time_dim=TIME, land_thresh=0.95):
    is_missing = np.isnan(da).all(dim)
    missing_freq = is_missing.mean(dim=time_dim)

    is_land_static = missing_freq > land_thresh

    mask_land = is_missing.copy()
    mask_land[:] = is_land_static

    mask_cloud = is_missing & (~mask_land.astype(bool))

    mask_cloud = mask_cloud.astype(np.float32).expand_dims(dim={dim: ['cloud_mask']})
    mask_land = mask_land.astype(np.float32).expand_dims(dim={dim: ['land_mask']})

    return mask_cloud, mask_land

# def get_masks(ds, dim):
#     masks = np.isfinite(ds).any(dim)
#     masks = masks.expand_dims(dim={dim: ['mask']})
#     return masks


def fill_na_minimum(ds, dims):
    mins = ds.min(dims)
    return ds.fillna(mins)


def positive_quantile(ds, quantile=0.01, dims=('Id', 'time', 'lat', 'lon')):
    quant = ds.where(ds > 0).quantile(quantile, dim=dims)
    x = np.maximum(ds, quant)
    return x


def quantile(ds, quantile=0.01, dims=('Id', 'time', 'lat', 'lon')):
    quant = ds.quantile(quantile, dim=dims)
    x = ds.where(ds > 0, other=quant)
    return x


def add_earth_mask_and_apply_logs(ds, dim='time', quantile=0.01):
    mask = np.isfinite(ds).any(dim)
    quant = ds.where(ds > 0).quantile(quantile, dim=('Id', 'time', 'lat', 'lon'))
    x = ds.where(ds > 0, other=quant)
    x = np.log(x)
    x = x.drop_vars(dim)
    return xr.concat([x, mask], dim=dim)


def dimension_len(ds, dim_name):
    if dim_name not in ds.sizes.keys():
        return 1
    return ds.sizes[dim_name]


def select_valid_data(da, threshold=0.2):
    dims = (dim_n for dim_n, s in da.sizes.items() if dim_n != 'Id')
    cube_dimension = np.product(np.product([s for dim_n, s in da.sizes.items() if dim_n != 'Id']))

    values_count = (~da.isnull()).sum(dim=dims)
    no_null_percent = values_count / cube_dimension
    valid_indices = (no_null_percent > threshold)!= 0
    return da.isel(Id=valid_indices)


def select_relative_valid_data(da, threshold=0.2):
    cube_dimension = np.product([s for dim_n, s in da.sizes.items() if dim_n != 'Id'])
    lats_tmp = set(da.lat.mean(axis=1).values)

    lats_indx = {l: da.lat.mean(axis=1) == l for l in lats_tmp}
    lats_groups = {l: da.sel(Id=ind) for l, ind in lats_indx.items()}
    max_coverage = {l: (~da_.mean(dim='Id').isnull()).sum().values / cube_dimension for l, da_ in
                    lats_groups.items()}

    valid_data = [select_valid_data(group, max_coverage[l] * threshold) for l, group in lats_groups.items()]
    ds_valid = xr.concat(valid_data, dim='Id')
    return ds_valid


def interpolate(ds, dims=(LAT, LON)):
    ds_interp = ds.copy()
    for band in tqdm(ds_interp.data_vars):
        for dim in dims:
            ds_interp[band] = ds_interp[band].interpolate_na(dim=dim, method="linear", use_coordinate=False)
    return ds_interp


def manhattan_distance(arr, p2):
    return np.sum(np.abs(arr-p2), axis=-1)


def radius_weights(sh):
    center = np.array(sh) // 2
    coords = it.product(*[list(range(i)) for i in sh])
    coords = np.array(list(coords))
    dist = manhattan_distance(coords, center)
    res = dist.reshape(sh)
    return res
# conv_sort = ["time", "lat", "lon"]


def compute_weights(ds):
    ds_cp = ds.copy()
    weights_arr = radius_weights(ds_cp.values.shape)
    conv_sort = list(ds_cp.dims)
    max_w = np.max(weights_arr)
    weights_arr = -weights_arr + max_w
    weights_arr[1] = weights_arr[1] * 3
    weights_arr = np.exp(weights_arr)

    weights = xr.DataArray(data=weights_arr, dims=conv_sort)
    return weights


def average(ds):
    coords_lat = ds.coords['lat'].mean(axis=1)
    coords_lon = ds.coords['lon'].mean(axis=1)
    if 'time' in ds.dims:
        coords_time = ds.coords['time'].mean(axis=1)
        ds_mean = ds.mean(dim=['lat', 'lon', 'time'], skipna=True, keep_attrs=True)
        ds_mean = xr.merge([ds_mean, coords_lat, coords_lon, coords_time])
    else:
        ds_mean = ds.mean(dim=['lat', 'lon'], skipna=True, keep_attrs=True)
        ds_mean = xr.merge([ds_mean, coords_lat, coords_lon])
    return ds_mean


def radius_weighted_average(ds, dims=(LAT, LON, TIME)):
    first_variable = list(ds.keys())[0]
    weights = compute_weights(ds.isel(Id=0)[first_variable])
    coords = [ds.coords[dim].mean(axis=1) for dim in dims]
    ds_weighted = ds.weighted(weights).mean(dim=dims, skipna=True, keep_attrs=True)
    ds_weighted = xr.merge([ds_weighted] + coords)
    return ds_weighted


def day_to_circle_x(day, period=365):
    angle = 2 * np.pi * day / period
    return np.cos(angle)


def day_to_circle_y(day, period=365):
    angle = 2 * np.pi * day / period
    return np.sin(angle)


def dms_to_decimal(dms):
    # Return as-is if already a number or NaN
    if pd.isna(dms) or isinstance(dms, (int, float)):
        return dms

    # Normalize all weird symbols and spacing
    dms = (str(dms)
           .replace("º", "°").replace("° ", "°")
           .replace("’", "'").replace("‘", "'")
           .replace("′", "'")
           .replace("″", "\"").replace("”", "\"").replace("“", "\"")
           .replace("''", "\"").replace("  ", " ")
           .strip())

    # Match pattern: degrees, minutes, seconds, direction
    match = re.match(r"(\d+)°\s*(\d+)'?\s*([\d\.]+)?\"?\s*([NSEW])?", dms)
    if not match:
        return dms  # leave unchanged if format unexpected

    deg, minutes, seconds, direction = match.groups()

    # Default seconds to 0 if missing
    seconds = seconds or 0

    # Convert to decimal degrees
    dec = float(deg) + float(minutes)/60 + float(seconds)/3600

    # Flip sign for South or West
    if direction in ['S', 'W']:
        dec = -dec

    return dec

def match_up(ids, lats, lons, times, region, lat_win=LAT_WINDOW, lon_win=LON_WINDOW, time_win=TIME_WINDOW,
             lat_th=LAT_TH, lon_th=LON_TH, time_th=TIME_TH):
    match_ups = []
    region = region.sortby(LAT).sortby(LON)
    for id_, lat, lon, date in tqdm(zip(ids, lats, lons, times), total=len(ids)):
        near_point = region.sel({TIME: date, LAT: lat, LON: lon}, method='nearest')

        near_date, near_lat, near_lon = near_point[TIME].values, near_point[LAT].values, near_point[LON].values
        if abs(near_date - date) < time_th and abs(near_lon - lon) < lon_th and abs(near_lat - lat) < lat_th:
            match = region.sel({TIME: slice(near_date - time_win, near_date + time_win),
                                    LAT: slice(near_lat - lat_win, near_lat + lat_win),
                                    LON: slice(near_lon - lon_win, near_lon + lon_win)})
            lats_coord = xr.DataArray([match[LAT].values], dims=['Id', LAT])
            lons_coord = xr.DataArray([match[LON].values], dims=['Id', LON])
            time_coord = xr.DataArray([match[TIME].values], dims=['Id', TIME])
            if 'depth' in list(match.coords):
                match = match.sel(depth=region.depth.min())
            match = match.assign_coords({LON: lons_coord, LAT: lats_coord, TIME: time_coord, 'Id': ('Id', [id_])})
            match_ups.append(match)
        # else:
        #     print(f"far away centered point({date},{lat},{lon}) with nearest ({near_date},{near_lat},{near_lon})")
    return match_ups