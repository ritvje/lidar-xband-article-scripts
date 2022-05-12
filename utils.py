"""Utility functions used in data processing."""
import re
import os
from datetime import datetime
import numpy as np
import pyart
import pyproj
import wradlib as wrl
import requests
import pandas as pd
from attrdict import AttrDict
from netCDF4 import Dataset
import xarray as xr
import struct


def query_Smartmet_station(
    fmisid, starttime, endtime, params, payload_params=None, url=None
):
    """Query SmartMet timeseries data from a weather station.

    Gets a timeseries from the station defined by `fmisid`.

    Parameters
    ----------
    fmisid : int
        FMISID for the station.
    starttime : datetime.datetime
        Start time for query.
    endtime : datetime.datetime
        End time for query.
    params : list
        List of queried parameters.
    payload_params : dict
        Extra parameters passed to query.
    url : str
        The base url to the query. Default is the internal FMI url.

    Returns
    -------
    df : pandas.DataFrame
        The results in a DataFrame, with UTC time of observations as
        index and parameter names as columns.

    """
    # Get weather information from SmartMet
    if url is None:
        url = "http://smartmet.fmi.fi/timeseries"
    if "timeseries" not in url:
        url += "/timeseries"

    if "utctime" not in params:
        params.append("utctime")

    if payload_params is None:
        payload_params = {}

    payload = {
        "producer": "observations_fmi",
        "fmisid": fmisid,  # Kumpula station
        "param": ",".join(params),
        "starttime": starttime,
        "endtime": endtime,
        "format": "json",
        "tz": "UTC",
        "precision": "double",
        **payload_params,
    }
    r = requests.get(url, params=payload)

    df = pd.DataFrame(r.json())

    df["time"] = pd.to_datetime(df["utctime"])

    # Transform all possible columns to numeric
    columns = [c for c in df.columns if "time" not in c]
    for column in columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    return df


def get_sigmet_file_list_by_task(
    path, file_regex="WRS([0-9]{12}).RAW([A-Z0-9]{4})", task_name=None
):
    """Generate a list of files by task.

    Parameters
    ----------
    path : str
        Path to data location
    file_regex : str
        Regex for matching filenames.
    task_name : str
        If given, only return files matching this task.

    Returns
    -------
    out : dict
        Dictionary of lists of filenames in `path`, with key giving the task name.

    """
    # Browse files
    prog = re.compile(file_regex.encode().decode("unicode_escape"))
    fullpath = os.path.abspath(path)

    data = []

    for (root, dirs, files) in os.walk(fullpath):
        addpath = root.replace(fullpath, "")

        for f in files:
            result = prog.match(f)

            if result is None:
                continue

            try:
                # Get task name from headers
                sf = pyart.io.sigmet.SigmetFile(os.path.join(root, f))
                task = (
                    sf.product_hdr["product_configuration"]["task_name"]
                    .decode()
                    .strip()
                )
                sf.close()
                if task_name is not None and task != task_name:
                    continue
                data.append([os.path.join(addpath, f), task])
            except (ValueError, OSError):
                continue
            except struct.error:
                print(f"Failed to read {f} with pyart!")
                continue

    df = pd.DataFrame(data, columns=["filename", "task_name"])
    out = {}
    for task, df_task in df.groupby("task_name"):
        out[task] = df_task["filename"].to_list()

    return out


def get_lidar_file_list_by_type(
    path,
    file_regex="WLS400s-113_([0-9_-]{19})_([a-z]+)_([0-9]+)_([0-9]+)m.nc",
    elev_angle=None,
    scan_type=None,
):
    """Generate a list of files by scan type and fixed angle.

    Parameters
    ----------
    path : str
        Path to data location
    file_regex : str
        Regex for matching filenames.
    elev_angle : float
        Filter by elevation angle.
    scan_type : str
        Filter by scan type.

    Returns
    -------
    out : dict
        Dictionary of lists of filenames in `path`, with key giving the scan type amd
        fixed angle.

    """
    # Browse files
    prog = re.compile(file_regex.encode().decode("unicode_escape"))
    fullpath = os.path.abspath(path)

    data = []

    for (root, _, files) in os.walk(fullpath):
        addpath = root.replace(fullpath, "")

        for f in files:
            result = prog.match(f)

            if result is None:
                continue

            try:
                lidar = Dataset(os.path.join(root, f), format="NETCDF4")
                sweep = lidar["sweep_group_name"][:][0]
                scan = lidar[sweep].variables["sweep_mode"][:]
                fixed_angle = lidar.variables["sweep_fixed_angle"][:][0]

                if (elev_angle is not None and fixed_angle != elev_angle) or (
                    scan_type is not None and scan != scan_type
                ):
                    continue

                data.append([os.path.join(addpath, f), scan, fixed_angle])
            except (ValueError, OSError):
                continue

    df = pd.DataFrame(data, columns=["filename", "scan_type", "fixed_angle"])
    out = {}
    for (stype, angle), df_task in df.groupby(["scan_type", "fixed_angle"]):
        out[(stype, angle)] = df_task["filename"].to_list()

    return out
