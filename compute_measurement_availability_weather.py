"""Compute measurement range for the instruments."""
import os
import sys
import argparse
import warnings
from datetime import datetime
from functools import partial
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from wradlib.io.xarray import CfRadial
import matplotlib as mlt
import seaborn as sns

mlt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dask
import dask.bag as db
import dask.array as da
import zarr
import cartopy.crs as ccrs
import pyart
import utils
import file_utils
import config as cfg
from radar_plotting import plotting
from radar_plotting import plotconfig as pcfg

warnings.simplefilter(action="ignore")

params = [
    "utctime",
    "stationname",
    "PRIO_PT10M_AVG",
    "CLHB_PT1M_INSTANT",
    "VIS_PT1M_AVG",
]
names = {
    "utctime": "Time (UTC)",
    "stationname": "Station name",
    "PRIO_PT10M_AVG": "Precipitation intensity (10min average) [mm h$^{-1}$]",
    "CLHB_PT1M_INSTANT": "Cloud base height [km]",
    "VIS_PT1M_AVG": "Horizontal visibility [km]",
}

station_fmisid = 100968
station_name = "Vantaa Helsinki-Vantaan lentoasema"


@ticker.FuncFormatter
def m2km_formatter(x, pos):
    return f"{x / 1000:.1f}"


def lidar_worker(
    ifn,
    zarr_array=None,
    datakey="radial_wind_speed",
    dsize=(120, 70),
    return_range_az=False,
):
    """Handle lidar files.

    Reads the requested data and grids it to Cartesian grid
    defined by `xgrid` and `ygrid`.

    Parameters
    ----------
    fn : str
        Filepath.
    datakey : str
        Key of the requested dataset.
    dsize : tuple
        Output array size.
    value_thr : float
        Values below this in data are masked out.

    Returns
    -------
    data : np.ma.array
        Data.

    """
    _, i, fn, _, _ = ifn
    try:
        cf2 = CfRadial(fn, flavour="Cf/Radial2", decode_times=False)
    except IOError:
        print(f"Failed to read {fn}")

        if zarr_array is not None:
            zarr_array[i, ...] = np.ones(dsize) * (-1)

        return np.ones(dsize) * np.nan
    sweep = list(cf2.keys())[0]
    data = np.ma.array(
        data=cf2[sweep][datakey].data, mask=np.zeros(cf2[sweep][datakey].data.shape)
    )
    data.set_fill_value(np.nan)
    np.ma.masked_where(cf2[sweep]["radial_wind_speed_status"] == 0, data, copy=False)

    if data.shape != dsize:
        print(f"File {fn} has size {data.shape}!")

        if zarr_array is not None:
            zarr_array[i, ...] = np.ones(dsize) * (-1)

        return np.ones(dsize) * np.nan

    if return_range_az:
        elev = np.nanmean(cf2[sweep].elevation.data)
        lonlatalt = np.array(
            [cf2[sweep].longitude.data.item(), cf2[sweep].latitude.data.item(), 35]
        )
        if any(np.isnan(lonlatalt)):
            return data
        return data, cf2[sweep].range.data, cf2[sweep].azimuth.data, elev, lonlatalt

    if zarr_array is not None:
        zarr_array[i, ...] = data.filled()
        return
    return data


def radar_worker(
    ifn,
    zarr_array=None,
    datakey="velocity",
    dsize=(360, 866),
    return_range_az=False,
):
    """Handle radar files.

    Reads the requested data and grids it to Cartesian grid
    defined by `xgrid` and `ygrid`.

    Parameters
    ----------
    fn : str
        Filepath.
    datakey : str
        Key of the requested dataset.
    dsize : tuple
        Output array size.
    value_thr : float
        Values below this in data are masked out.

    Returns
    -------
    data : np.ma.array
        Data.

    """
    _, i, _, fn, _ = ifn
    try:
        radar = pyart.io.read_sigmet(fn, include_fields=[datakey])
    except (ValueError, OSError, IOError, IndexError):
        print(f"Failed to read {fn}")

        if zarr_array is not None:
            zarr_array[i, ...] = np.ones(dsize) * (-1)

        return np.ones(dsize) * np.nan
    data = radar.get_field(0, datakey)
    data.set_fill_value(np.nan)

    if data.shape != dsize:
        print(f"File {fn} has size {data.shape}!")

        if zarr_array is not None:
            zarr_array[i, ...] = np.ones(dsize) * (-1)

        return np.ones(dsize) * np.nan

    if return_range_az:
        elev = radar.fixed_angle["data"][0]
        lonlatalt = np.array(
            [radar.longitude["data"][0], radar.latitude["data"][0], 35]
        )
        return data, radar.range["data"], radar.azimuth["data"], elev, lonlatalt

    if zarr_array is not None:
        zarr_array[i, ...] = data.filled()
        return
    return data.filled()


def main(
    startdate, enddate, xband_task, outpath, valid_pct_thr=0.05, only_read_data=True
):
    # Read config
    lidar_cfg = cfg.LIDAR_INFO["vaisala"]
    basepath = cfg.MWSA_DATA_PATH
    lidar_dsize = (120, 70)
    radar_dsize = (360, 866)

    get_xband_files = partial(
        utils.get_sigmet_file_list_by_task,
        task_name=xband_task,
    )

    # Util func to get date from xband path
    def xband_date(f):
        return datetime.strptime(os.path.basename(f).split(".")[0], "WRS%y%m%d%H%M%S")

    # Loop over days and get files
    dateinterval = pd.date_range(startdate, enddate, freq="D")

    if not only_read_data:
        COMMON_FILES = []

        def get_files(day):
            res = []
            path = os.path.join(basepath, f"{day:%Y/%m/%d}")

            # Get lidar files for the day
            lidar_files = file_utils.find_matching_filenames(
                path,
                lidar_cfg["filepattern"],
                lidar_cfg["timepattern"],
            )
            # Get xband files for the given task and add to dictionary with time as key
            xband_fn_corr_tasks = get_xband_files(path)
            if len(xband_fn_corr_tasks.keys()) == 0:
                return []
            xband_fn_corr_tasks = xband_fn_corr_tasks[
                list(xband_fn_corr_tasks.keys())[0]
            ]
            xband_files = {xband_date(f): path + f for f in xband_fn_corr_tasks}

            # For each lidar file, get closest xband file
            for lidar_time, lidar_fn in sorted(lidar_files.items()):
                _, xband_fn, td = file_utils.find_closest_file(lidar_time, xband_files)
                res.append((0, lidar_time, lidar_fn, xband_fn, td))
            return res

        # Get common dates for the intervals
        res = []
        for day in dateinterval:
            res.append(dask.delayed(get_files)(day))

        res = dask.compute(
            *res, num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER
        )
        for r in res:
            COMMON_FILES.extend(r)

        df = pd.DataFrame(
            COMMON_FILES, columns=["index", "time", "lidarfn", "xbandfn", "dt"]
        )
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        # Set running index
        df["index"] = np.arange(len(df))
        # Save to csv to just read later
        df.to_csv(outpath / f"files_{startdate:%Y%m%d}_{enddate:%Y%m%d}.csv")
    else:
        # Read data that was processed before
        df = pd.read_csv(
            outpath / f"files_{startdate:%Y%m%d}_{enddate:%Y%m%d}.csv",
            index_col=0,
            parse_dates=[
                0,
            ],
        )
    logging.info(f"Found {len(df)} lidar & x-band files!")

    # Load weather data
    tol = pd.Timedelta(f"{args.tol} minute")
    wdf = utils.query_Smartmet_station(
        station_fmisid,
        startdate,
        enddate.replace(hour=23, minute=59),
        params,
    )
    wdf.set_index("time", inplace=True)

    # Drop empty rows (if no observations given for some time)
    drop_cols = [c for c in wdf.columns if str(tol.seconds // 60) in c]
    wdf.dropna(how="all", subset=drop_cols, inplace=True)

    if args.tol > 0:
        # Merge dataframes
        dff = pd.merge_asof(
            left=wdf,
            right=df,
            right_index=True,
            left_index=True,
            tolerance=tol,
            direction="backward",
        )
    else:
        dff = df

    # Initialize zarr arrays for storing output values
    mode = "r" if only_read_data else "w"
    lidar_synchronizer = zarr.ProcessSynchronizer(str(outpath / f"lidar.sync"))
    lidar_data_output = zarr.open_array(
        str(outpath / f"lidar_{startdate:%Y%m}_{enddate:%Y%m}.zarr"),
        mode=mode,
        shape=(len(df), *lidar_dsize),
        chunks=(1000, *lidar_dsize),
        synchronizer=lidar_synchronizer,
    )
    xband_synchronizer = zarr.ProcessSynchronizer(str(outpath / f"xband.sync"))
    xband_data_output = zarr.open_array(
        str(outpath / f"radar_{startdate:%Y%m}_{enddate:%Y%m}.zarr"),
        mode=mode,
        shape=(len(df), *radar_dsize),
        chunks=(500, *radar_dsize),
        synchronizer=xband_synchronizer,
    )

    # Get dimensions of data
    for tp in df.itertuples():
        r = lidar_worker(tp, return_range_az=True)
        if len(r) == 5:
            lidar_rr = r[1]
            break

    for tp in df.itertuples():
        r = radar_worker(tp, return_range_az=True)
        if len(r) == 5:
            xband_rr = r[1]
            break

    xband_rr_mask = xband_rr <= np.max(lidar_rr)

    # Run calculation as dask bag
    # The output from each worker is an array of same size,
    # so it's handy to stack the results into dask array
    VALID_PCT = 0.05
    with dask.config.set(num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER):
        if not only_read_data:
            # X-band
            bx = db.from_sequence(df.itertuples())
            bx.map(radar_worker, zarr_array=xband_data_output).compute()

    with dask.config.set(num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER):
        if not only_read_data:
            # Lidar
            bl = db.from_sequence(df.itertuples())
            bl.map(lidar_worker, zarr_array=lidar_data_output).compute()

    with dask.config.set(num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER):
        lidar_arr = da.from_zarr(lidar_data_output)
        xband_arr = da.from_zarr(xband_data_output)
        # Remove times when either data is invalid
        lidar_valid_mask = da.count_nonzero(lidar_arr == -1, axis=[1, 2]).compute()
        xband_valid_mask = da.count_nonzero(xband_arr == -1, axis=[1, 2]).compute()
        total_valid_mask = ~(
            (lidar_valid_mask == (lidar_arr.shape[1] * lidar_arr.shape[2]))
            | (xband_valid_mask == (xband_arr.shape[1] * xband_arr.shape[2]))
        )

        # Calculate blockages in full data
        # Number of total valid bins in arrays & blockages
        larr = lidar_arr[total_valid_mask]
        xarr = xband_arr[total_valid_mask]
        n_valid_in_bin_lidar = da.sum(da.isfinite(larr), axis=0).compute()

        blockage_lidar = (
            np.sum(n_valid_in_bin_lidar, axis=1) / larr.shape[0] / larr.shape[2]
            < VALID_PCT
        )
        n_valid_in_bin_xband = da.sum(da.isfinite(xarr), axis=0).compute()
        blockage_xband = (
            np.sum(n_valid_in_bin_xband, axis=1) / xarr.shape[0] / xarr.shape[2]
            < VALID_PCT
        )

        # Filter with weather data
        if args.tol == 1:
            wvar = "CLHB_PT1M_INSTANT"
            limits = np.array(
                [0, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]
            )
            lidar_result, xband_result = compute_measurement_ranges(
                dff,
                wvar,
                limits,
                lidar_arr,
                xband_arr,
                total_valid_mask,
                xband_rr_mask,
                blockage_lidar,
                blockage_xband,
                lidar_rr,
                xband_rr,
            )
            plot_meas_ranges_log_scale(
                lidar_result,
                xband_result,
                wvar,
                cbar_formatter=m2km_formatter,
            )

            wvar = "VIS_PT1M_AVG"
            limits = np.arange(0, 80e3, 5000)

            lidar_result, xband_result = compute_measurement_ranges(
                dff,
                wvar,
                limits,
                lidar_arr,
                xband_arr,
                total_valid_mask,
                xband_rr_mask,
                blockage_lidar,
                blockage_xband,
                lidar_rr,
                xband_rr,
            )
            plot_meas_ranges(
                lidar_result,
                xband_result,
                wvar,
                cbar_formatter=m2km_formatter,
            )

        elif args.tol == 10:
            wvar = "PRIO_PT10M_AVG"
            limits = np.arange(0, 4.1, 0.25)

            lidar_result, xband_result = compute_measurement_ranges(
                dff,
                wvar,
                limits,
                lidar_arr,
                xband_arr,
                total_valid_mask,
                xband_rr_mask,
                blockage_lidar,
                blockage_xband,
                lidar_rr,
                xband_rr,
            )
            plot_meas_ranges(
                lidar_result,
                xband_result,
                wvar,
                cbar_formatter=None,
            )
        elif args.tol == 0:
            lidar_result, xband_result = compute_measurement_ranges_wo_weather(
                lidar_arr,
                xband_arr,
                total_valid_mask,
                xband_rr_mask,
                blockage_lidar,
                blockage_xband,
                lidar_rr,
                xband_rr,
            )


def compute_measurement_ranges_wo_weather(
    lidar_arr,
    xband_arr,
    total_valid_mask,
    xband_rr_mask,
    blockage_lidar,
    blockage_xband,
    lidar_rr,
    xband_rr,
):
    lidar_result = pd.DataFrame(index=lidar_rr, columns=["pct"])
    xband_result = pd.DataFrame(index=xband_rr, columns=["pct"])

    # Remove unvalid times and for xband also ranges further than lidar coverage
    larr = lidar_arr[total_valid_mask, ...]
    xarr = xband_arr[total_valid_mask, ...]

    # LIDAR
    n_valid_in_bin_lidar_1 = da.sum(da.isfinite(larr), axis=0).compute()

    rr_count_lidar_1 = np.sum(n_valid_in_bin_lidar_1[~blockage_lidar], axis=0)
    n_valid_scans_lidar_1 = (~da.all(da.isnan(larr), axis=(1, 2)).compute()).sum()
    rr_pct_lidar_1 = rr_count_lidar_1 / n_valid_scans_lidar_1 / (~blockage_lidar).sum()
    lidar_result["pct"] = rr_pct_lidar_1

    # X-BAND
    n_valid_in_bin_xband_2 = da.sum(da.isfinite(xarr), axis=0).compute()

    rr_count_xband_2 = np.sum(n_valid_in_bin_xband_2[~blockage_xband], axis=0)
    n_valid_scans_xband_2 = (~da.all(da.isnan(xarr), axis=(1, 2)).compute()).sum()
    rr_pct_xband_2 = rr_count_xband_2 / n_valid_scans_xband_2 / (~blockage_xband).sum()
    xband_result["pct"] = rr_pct_xband_2

    lidar_csvfn = Path(f"measurement_ranges_lidar.csv")
    lidar_result.to_csv(outpath / lidar_csvfn)
    xband_csvfn = Path(f"measurement_ranges_xband.csv")
    xband_result.to_csv(outpath / xband_csvfn)
    return lidar_result, xband_result


def compute_measurement_ranges(
    dff,
    wvar,
    limits,
    lidar_arr,
    xband_arr,
    total_valid_mask,
    xband_rr_mask,
    blockage_lidar,
    blockage_xband,
    lidar_rr,
    xband_rr,
):
    lidar_result = pd.DataFrame(index=lidar_rr, columns=limits)
    xband_result = pd.DataFrame(index=xband_rr, columns=limits)

    groups = dff.groupby(pd.cut(dff[wvar], limits))

    for limits, df in groups:

        if df.empty:
            continue

        weather_idx = df["index"].dropna().astype(int).values
        total_idx = np.intersect1d(np.where(total_valid_mask)[0], weather_idx)
        total_mask = np.zeros_like(total_valid_mask)
        total_mask[total_idx] = 1

        # Remove unvalid times and for xband also ranges further than lidar coverage
        larr = lidar_arr[total_valid_mask, ...]
        xarr = xband_arr[total_valid_mask, ...]  # [..., xband_rr_mask]

        # Filter with weather
        larr = lidar_arr[total_mask, ...]
        xarr = xband_arr[total_mask, ...]  # [..., xband_rr_mask]

        # LIDAR
        n_valid_in_bin_lidar_1 = da.sum(da.isfinite(larr), axis=0).compute()

        rr_count_lidar_1 = np.sum(n_valid_in_bin_lidar_1[~blockage_lidar], axis=0)
        n_valid_scans_lidar_1 = (~da.all(da.isnan(larr), axis=(1, 2)).compute()).sum()
        rr_pct_lidar_1 = (
            rr_count_lidar_1 / n_valid_scans_lidar_1 / (~blockage_lidar).sum()
        )
        lidar_result[limits.right] = rr_pct_lidar_1

        # X-BAND
        n_valid_in_bin_xband_2 = da.sum(da.isfinite(xarr), axis=0).compute()

        rr_count_xband_2 = np.sum(n_valid_in_bin_xband_2[~blockage_xband], axis=0)
        n_valid_scans_xband_2 = (~da.all(da.isnan(xarr), axis=(1, 2)).compute()).sum()
        rr_pct_xband_2 = (
            rr_count_xband_2 / n_valid_scans_xband_2 / (~blockage_xband).sum()
        )
        xband_result[limits.right] = rr_pct_xband_2

    lidar_csvfn = Path(f"measurement_ranges_lidar_{wvar}_{limits.right:.0f}.csv")
    lidar_result.to_csv(outpath / lidar_csvfn)
    xband_csvfn = Path(f"measurement_ranges_xband_{wvar}_{limits.right:.0f}.csv")
    xband_result.to_csv(outpath / xband_csvfn)
    return lidar_result, xband_result


def plot_meas_ranges(
    lidar_result,
    xband_result,
    wvar,
    cmap="viridis",
    cbar_formatter=None,
):
    # Plot results
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [1, 1]},
        figsize=(9, 7),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = sns.color_palette(cmap, as_cmap=True)
    norm = mlt.colors.Normalize(
        vmin=lidar_result.columns.min(),
        vmax=lidar_result.columns.max(),
    )
    lw = 1.5

    for upperlim in lidar_result.columns:
        axes[0, 0].plot(
            lidar_result.index.values * 1e-3,
            lidar_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )
        axes[1, 0].plot(
            xband_result.index.values * 1e-3,
            xband_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        label=names[wvar],
        ax=axes,
        aspect=50,
        orientation="vertical",
        shrink=0.6,
        extend="max",
        # location="top",
        format=cbar_formatter,
    )
    cbar.set_label(label=names[wvar], weight="bold")
    # cbar.ax.xaxis.set_label_position("bottom")
    # cbar.ax.xaxis.set_ticks_position("bottom")

    axes[-1, 0].set_xlabel("Range [km]")
    titles = ["(a) Lidar", "(b) X-band radar"]
    for ax, title in zip(axes.flat, titles):
        ax.set_ylabel("Fraction")
        # ax.set_ylim(())

        # Set axis limits
        ax.set_xlim((0, 15))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_ylim((0, 1.05))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.grid(which="both", alpha=0.5)
        ax.set_title(title, y=1.0)

    # fig.suptitle(f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}")

    fig.savefig(
        outpath / f"measurement_range_{wvar}_{np.max(lidar_result.columns):.0f}_"
        f"{startdate:%Y%m%d}_{enddate:%Y%m%d}.pdf",
        dpi=600,
        bbox_inches="tight",
    )


def plot_meas_ranges_log_scale(
    lidar_result,
    xband_result,
    wvar,
    cmap="viridis",
    cbar_formatter=None,
):
    # Plot results
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [1, 1]},
        figsize=(9, 7),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = sns.color_palette(cmap, as_cmap=True)
    norm = mlt.colors.LogNorm(
        vmin=lidar_result.columns[1],
        vmax=lidar_result.columns[-1],
    )
    lw = 1.5

    for upperlim in lidar_result.columns[1:]:
        axes[0, 0].plot(
            lidar_result.index.values * 1e-3,
            lidar_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )
        axes[1, 0].plot(
            xband_result.index.values * 1e-3,
            xband_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        label=names[wvar],
        ax=axes,
        aspect=50,
        orientation="vertical",
        shrink=0.6,
        extend="max",
        # location="top",
        format=cbar_formatter,
    )
    cbar.set_label(label=names[wvar], weight="bold")
    cbar.set_ticks(lidar_result.columns)
    #     cbar.set_ticklabels(lidar_result.columns)
    # cbar.ax.xaxis.set_label_position("bottom")
    # cbar.ax.xaxis.set_ticks_position("bottom")

    axes[-1, 0].set_xlabel("Range [km]")
    titles = ["(a) Lidar", "(b) X-band radar"]
    for ax, title in zip(axes.flat, titles):
        ax.set_ylabel("Fraction")
        # ax.set_ylim(())

        # Set axis limits
        ax.set_xlim((0, 15))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_ylim((0, 1.05))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.grid(which="both", alpha=0.5)
        ax.set_title(title, y=1.0)

    # fig.suptitle(f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}")

    fig.savefig(
        outpath / f"measurement_range_{wvar}_{np.max(lidar_result.columns):.0f}_"
        f"{startdate:%Y%m%d}_{enddate:%Y%m%d}.pdf",
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("startdate", type=str, help="the startdate (YYYYmm)")
    argparser.add_argument("enddate", type=str, help="the enddate (YYYYmm)")
    argparser.add_argument(
        "--task-name", type=str, default="WND-03", help="X-band task name"
    )
    argparser.add_argument("--outpath", type=str, default=".", help="Output path")
    argparser.add_argument(
        "--only-read",
        action="store_true",
        default=False,
        help="Read data from previously stored",
    )
    argparser.add_argument(
        "--tol",
        type=int,
        default=1,
        help="Tolerance for merging weather observations, minutes",
    )

    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m%d")
    enddate = datetime.strptime(args.enddate, "%Y%m%d")

    outpath = Path(args.outpath)

    # Set style file
    plt.style.use(cfg.STYLE_FILE)

    logging.basicConfig(level=logging.INFO)

    main(startdate, enddate, args.task_name, outpath, only_read_data=args.only_read)
