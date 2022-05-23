"""Utility functions for handling files.

Includes functions
- get_closest_radar_time
- find_closest_file
- get_files
- find_matching_filenames
- parse_time_from_filename
- get_sigmet_file_list_by_task
- get_lidar_file_list_by_type

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
import re
import numpy as np
from datetime import datetime, timedelta
import pyart
from netCDF4 import Dataset
import struct
import pandas as pd


def get_closest_radar_time(lidar_time, start_secs):
    """Get the closest starting time of radar scan to lidar_time.

    Parameters
    ----------
    lidar_time : datetime.datetime
        Time.
    start_secs : list
        List of seconds from previous 5-minute time when scans start.

    Returns
    -------
    file_time : datetime.datetime
        The 5-minute floored time of the radar scan.
    ind : int
        Index of the scan in start_secs.

    """
    quarter = lidar_time - timedelta(
        minutes=lidar_time.minute % 15,
        seconds=lidar_time.second,
        microseconds=lidar_time.microsecond,
    )
    # start time for the tasks, from the starting 5 minute interval
    # start_secs = [129, 142, 142]
    start_times = [
        quarter + timedelta(seconds=(i * 300 + t)) for i, t in enumerate(start_secs)
    ]
    time_diff = [
        (st - lidar_time).seconds if st > lidar_time else (lidar_time - st).seconds
        for st in start_times
    ]
    min_ind = int(np.argmin(time_diff))
    return quarter + timedelta(seconds=((min_ind * 300))), min_ind


def find_closest_file(time, filedict):
    """Find the file that is closest in time to the given time.

    Parameters
    ----------
    time : datetime.datetime
        Datetime that files are compared to.
    filedict : dict
        Dictionary of datetime: filepath items, as returned by
        `find_matching_filenames`.

    Returns
    -------
    min_time : datetime.datetime
        The time of the file.
    min_fn : str
        Path to the file.

    """
    time_diff = [
        (st - time).total_seconds() if st > time else (time - st).total_seconds()
        for st in filedict
    ]
    min_ind = int(np.argmin(time_diff))
    min_time = list(filedict)[min_ind]
    min_fn = filedict[min_time]

    return min_time, min_fn, time_diff[min_ind]


def get_files(ds_config, date):
    """Get radar file path for given date.

    Parameters
    ----------
    ds_config : dict
        Datasource config for the radar.
    date : datetime.datetime
        The date that files are searched for.

    Returns
    -------
    filenames : dict
        Dict of date: filepath pairs.

    """
    if ds_config["path_fmt"] is not None:
        path_fmt = date.strftime(ds_config["path_fmt"])
    else:
        path_fmt = ""
    search_path = os.path.join(ds_config["root_path"], path_fmt)
    pattern = ds_config["fn_pattern"] + r"\." + ds_config["fn_ext"]
    filenames = find_matching_filenames(
        search_path, pattern, ds_config["fn_ts_pattern"], allow_multiple=True
    )
    return filenames


def find_matching_filenames(
    path, pattern, date_pattern="%Y%m%d%H%M", allow_multiple=False
):
    """Recursively search the given path for filenames matching the given pattern.

    Author: Seppo Pulkkinen.

    Parameters
    ----------
    path : str
        The path to search from.
    pattern : str
        Regular expression against which the file names are matched.
    date_pattern : str
        Time stamp pattern in the file names, see the documentation of the datetime
        module.
    allow_multiple : bool
        If True, allow multiple file names to match the same pattern.

    Returns
    -------
    out : dict
        The matching file names. Keys are datetime objects and values are either
        strings or lists of strings depending on the value of the allow_multiple
        argument.

    """
    fn_dict = {}

    po = re.compile(pattern)
    for f in os.walk(path):
        dirpath = f[0]
        fns = f[2]
        for fn in fns:
            m = po.match(fn)
            if m is not None:
                d = datetime.strptime(m.groups()[0], date_pattern)
                if not allow_multiple:
                    fn_dict[d] = os.path.join(dirpath, fn)
                else:
                    if d not in fn_dict.keys():
                        fn_dict[d] = []
                    fn_dict[d].append(os.path.join(dirpath, fn))

    return fn_dict


def parse_time_from_filename(fn, pattern, timepattern, group_idx=0):
    """Parse time from filename."""
    po = re.compile(pattern)
    match = po.match(fn)
    if match is not None:
        return datetime.strptime(match.groups()[group_idx], timepattern)
    else:
        return None


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
