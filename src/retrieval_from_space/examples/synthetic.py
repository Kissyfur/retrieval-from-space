from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _field(shape, base, time_signal, lat_signal, lon_signal, noise, rng):
    time_len, lat_len, lon_len = shape
    t = np.linspace(0.0, 1.0, time_len)[:, None, None]
    y = np.linspace(-1.0, 1.0, lat_len)[None, :, None]
    x = np.linspace(-1.0, 1.0, lon_len)[None, None, :]
    values = base + time_signal * t + lat_signal * y + lon_signal * x
    return values + rng.normal(0.0, noise, size=shape)


def create_synthetic_example(
    output_dir: str | Path = "examples/synthetic/generated",
    n_observations: int = 72,
    seed: int = 42,
) -> dict[str, Path]:
    """Create target observations and Copernicus-like local NetCDF products.

    The files intentionally use `latitude`/`longitude` coordinate names so the
    normal download stage has to normalize them to the internal `lat`/`lon`
    convention.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    source_dir = output_dir / "copernicus_like"
    source_dir.mkdir(parents=True, exist_ok=True)

    times = pd.date_range("2020-01-01", periods=30, freq="D")
    latitudes = np.linspace(39.5, 40.5, 21)
    longitudes = np.linspace(1.0, 2.0, 21)
    shape = (len(times), len(latitudes), len(longitudes))

    rrs443 = np.clip(_field(shape, 0.012, 0.005, 0.002, -0.001, 0.0004, rng), 0.001, None)
    rrs555 = np.clip(_field(shape, 0.018, -0.003, 0.001, 0.002, 0.0004, rng), 0.001, None)
    chl = np.clip(_field(shape, 1.4, 0.8, 0.3, -0.2, 0.05, rng), 0.05, None)

    reflectance = xr.Dataset(
        {
            "RRS443": (("time", "latitude", "longitude"), rrs443),
            "RRS555": (("time", "latitude", "longitude"), rrs555),
            "CHL": (("time", "latitude", "longitude"), chl),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        attrs={"title": "Synthetic Copernicus-like reflectance product"},
    )

    temperature = _field(shape, 18.0, 4.0, -1.5, 0.7, 0.2, rng)
    salinity = _field(shape, 37.5, 0.2, 0.5, -0.2, 0.03, rng)
    current_u = _field(shape, 0.05, 0.03, 0.02, -0.01, 0.01, rng)
    current_v = _field(shape, -0.02, 0.01, -0.03, 0.02, 0.01, rng)
    physics = xr.Dataset(
        {
            "temp": (("time", "latitude", "longitude"), temperature),
            "sal": (("time", "latitude", "longitude"), salinity),
            "uo": (("time", "latitude", "longitude"), current_u),
            "vo": (("time", "latitude", "longitude"), current_v),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        attrs={"title": "Synthetic Copernicus-like physics product"},
    )

    reflectance_path = source_dir / "synthetic_reflectance_source.nc"
    physics_path = source_dir / "synthetic_physics_source.nc"
    reflectance.to_netcdf(reflectance_path)
    physics.to_netcdf(physics_path)

    obs_rows = []
    valid_time_indices = np.arange(2, len(times) - 2)
    valid_lat_indices = np.arange(2, len(latitudes) - 2)
    valid_lon_indices = np.arange(2, len(longitudes) - 2)
    for obs_id in range(n_observations):
        ti = int(rng.choice(valid_time_indices))
        yi = int(rng.choice(valid_lat_indices))
        xi = int(rng.choice(valid_lon_indices))
        signal = (
            650.0 * rrs443[ti, yi, xi]
            + 170.0 * rrs555[ti, yi, xi]
            + 0.85 * chl[ti, yi, xi]
            + 0.22 * temperature[ti, yi, xi]
            - 0.08 * salinity[ti, yi, xi]
            + 20.0 * np.hypot(current_u[ti, yi, xi], current_v[ti, yi, xi])
        )
        target_value = signal + rng.normal(0.0, 0.15)
        obs_rows.append(
            {
                "Id": obs_id,
                "lat": latitudes[yi],
                "lon": longitudes[xi],
                "time": times[ti].date().isoformat(),
                "target_value": target_value,
                "station": f"S{obs_id % 4}",
                "sampling_depth": float((obs_id % 3) * 5),
                "field_temperature": float(temperature[ti, yi, xi] + rng.normal(0.0, 0.1)),
            }
        )

    targets = pd.DataFrame(obs_rows)
    target_path = output_dir / "targets.csv"
    targets.to_csv(target_path, index=False)

    return {
        "target_table": target_path,
        "reflectance_product": reflectance_path,
        "physics_product": physics_path,
    }
