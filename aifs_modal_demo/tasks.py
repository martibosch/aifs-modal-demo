"""Pipeline tasks for AIFS experiments."""

import datetime
import tempfile
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr
from meteora import clients, units, utils
from shapely import geometry


def meteoswiss_stationbench(start_date, end_date, lat_slice, lon_slice, dst_filepath):
    """Download MeteoSwiss station observations and save as stationbench-ready netCDF.

    Parameters
    ----------
    start_date, end_date : datetime-like
        Observation window.
    lat_slice, lon_slice : tuple of float
        Latitude and longitude bounding box.
    dst_filepath : path-like
        Path where the output netCDF file will be written.
    """
    dst_filepath = Path(dst_filepath)
    dst_filepath.parent.mkdir(parents=True, exist_ok=True)

    client = clients.MeteoSwissClient(
        region=geometry.box(*lon_slice, *lat_slice), crs="epsg:4326"
    )
    ts_df = client.get_ts_df(
        variables=["temperature", "wind_speed"],
        start=start_date.replace(tzinfo=None),
        end=end_date.replace(tzinfo=None),
    )
    ts_df = units.convert_units(ts_df, {"temperature": "K"})
    ts_ds = utils.long_to_stationbench(ts_df, client.stations_gdf)
    ts_ds.to_netcdf(dst_filepath)


def era5_stationbench(start_date, lead_time, dst_filepath):
    """Fetch ERA5 reanalysis from CDS API and save as stationbench-ready netCDF.

    Downloads 2 m temperature and 10 m wind components from ERA5 single-level reanalysis
    for the valid-time window defined by *start_date* and *lead_time*, computes 10 m
    wind speed, and writes a stationbench-ready dataset with dimensions
    `(time, prediction_timedelta, latitude, longitude)`.

    Parameters
    ----------
    start_date : datetime.datetime
        Forecast initialisation time (with or without tzinfo).
    lead_time : int
        Lead time in hours.
    dst_filepath : str or Path
        Destination netCDF file path.
    """
    dst_filepath = Path(dst_filepath)
    dst_filepath.parent.mkdir(parents=True, exist_ok=True)

    start_date_naive = start_date.replace(tzinfo=None)
    init_time = np.datetime64(start_date_naive, "ns")
    valid_dts = [
        start_date_naive + datetime.timedelta(hours=h)
        for h in range(6, lead_time + 1, 6)
    ]
    valid_times = np.array([np.datetime64(dt, "ns") for dt in valid_dts])
    lead_times = valid_times - init_time

    years = sorted({dt.strftime("%Y") for dt in valid_dts})
    months = sorted({dt.strftime("%m") for dt in valid_dts})
    days = sorted({dt.strftime("%d") for dt in valid_dts})
    hours = sorted({dt.strftime("%H:00") for dt in valid_dts})

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = Path(tmp_dir) / "foo.nc"
        cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(tmp_filepath),
        )

        ds = xr.open_dataset(tmp_filepath)
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds = ds.sel({time_dim: valid_times})

        # rename variables to short names if needed
        rename_map = {}
        if "2m_temperature" in ds:
            rename_map["2m_temperature"] = "t2m"
        if "10m_u_component_of_wind" in ds:
            rename_map["10m_u_component_of_wind"] = "u10"
        if "10m_v_component_of_wind" in ds:
            rename_map["10m_v_component_of_wind"] = "v10"
        if rename_map:
            ds = ds.rename(rename_map)

        # Build the dataset directly in stationbench format instead of
        # attaching prediction_timedelta as a coordinate on the source time axis.
        era5_ds = xr.Dataset(
            {
                "2t": (
                    ["time", "prediction_timedelta", "latitude", "longitude"],
                    ds["t2m"].values[np.newaxis],
                ),
                "10si": (
                    ["time", "prediction_timedelta", "latitude", "longitude"],
                    np.sqrt(ds["u10"].values ** 2 + ds["v10"].values ** 2)[np.newaxis],
                ),
            },
            coords={
                "time": [init_time],
                "prediction_timedelta": lead_times,
                "latitude": ds["latitude"].values,
                "longitude": ds["longitude"].values,
            },
        )

        era5_ds.to_netcdf(dst_filepath)
