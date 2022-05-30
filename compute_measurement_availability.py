"""Compute measurement range for the instruments.

Produces
- fraction of available measurements as function of range
- Cartesian image of fraction of available measurements

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
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

mlt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dask
import dask.bag as db
import dask.array as da
import zarr
import pyart

import utils
import file_utils
import config as cfg
from radar_plotting import plotting

warnings.simplefilter(action="ignore")


def lidar_worker(
    ifn,
    zarr_array=None,
    datakey="radial_wind_speed",
    dsize=(120, 70),
    return_range_az=False,
    altitude=35,
):
    """Handle lidar files.

    Reads the requested data and returns it. Returns a nan array, if reading fails.

    Parameters
    ----------
    ifn : tuple
        Tuple of (index, filepath).
        Filepath.
    zarr_array : zarr.array
        Array where the read data is written to in `index` place.
    datakey : str
        Key of the requested dataset.
    dsize : tuple
        Output array size.
    return_range_az : bool
        If true, don't write data but return data, range, azimuth, elevation, lonlatalt.
    altitude : float
        The altitude of the instrument, returned in lonlatalt.
    value_thr : float
        Values below this in data are masked out.

    Returns
    -------
    data : np.ma.array
        Data.
    range : np.ndarray
        Range bins
    azimuth : np.ndarray
        Azimuth values
    elev : float
        Elevation angle
    lonlatalt : tuple
        Longitude, latitude read from file

    """
    i, fn = ifn
    try:
        cf2 = CfRadial(fn, flavour="Cf/Radial2", decode_times=False)
    except IOError:
        print(f"Failed to read {fn}")
        return np.ones(dsize) * np.nan
    sweep = list(cf2.keys())[0]
    data = np.ma.array(
        data=cf2[sweep][datakey].data, mask=np.zeros(cf2[sweep][datakey].data.shape)
    )
    data.set_fill_value(np.nan)
    np.ma.masked_where(cf2[sweep]["radial_wind_speed_status"] == 0, data, copy=False)

    if data.shape != dsize:
        print(f"File {fn} has size {data.shape}!")
        return np.ones(dsize) * np.nan

    if return_range_az:
        elev = np.nanmean(cf2[sweep].elevation.data)
        lonlatalt = np.array(
            [
                cf2[sweep].longitude.data.item(),
                cf2[sweep].latitude.data.item(),
                altitude,
            ]
        )
        if any(np.isnan(lonlatalt)):
            return data
        return data, cf2[sweep].range.data, cf2[sweep].azimuth.data, elev, lonlatalt

    if zarr_array is not None:
        zarr_array[i, ...] = data.filled()
        # del data, cf2
        return
    return data


def radar_worker(
    ifn,
    zarr_array=None,
    datakey="velocity",
    dsize=(360, 866),
    return_range_az=False,
    altitude=35,
):
    """Handle radar files.

    Reads the requested data and returns it. Returns a nan array, if reading fails.

    Parameters
    ----------
    ifn : tuple
        Tuple of (index, filepath).
        Filepath.
    zarr_array : zarr.array
        Array where the read data is written to in `index` place.
    datakey : str
        Key of the requested dataset.
    dsize : tuple
        Output array size.
    return_range_az : bool
        If true, don't write data but return data, range, azimuth, elevation, lonlatalt.
    altitude : float
        The altitude of the instrument, returned in lonlatalt.

    Returns
    -------
    data : np.ma.array
        Data.
    range : np.ndarray
        Range bins
    azimuth : np.ndarray
        Azimuth values
    elev : float
        Elevation angle
    lonlatalt : tuple
        Longitude, latitude read from file

    """
    i, fn = ifn
    try:
        radar = pyart.io.read_sigmet(fn, include_fields=[datakey])
    except (ValueError, OSError, IOError, IndexError):
        print(f"Failed to read {fn}")
        return np.ones(dsize) * np.nan
    data = radar.get_field(0, datakey)
    data.set_fill_value(np.nan)

    if data.shape != dsize:
        print(f"File {fn} has size {data.shape}!")
        return np.ones(dsize) * np.nan

    if return_range_az:
        elev = radar.fixed_angle["data"][0]
        lonlatalt = np.array(
            [radar.longitude["data"][0], radar.latitude["data"][0], altitude]
        )
        return data, radar.range["data"], radar.azimuth["data"], elev, lonlatalt

    if zarr_array is not None:
        zarr_array[i, ...] = data.filled()
        # del data, cf2
        return
    return data.filled()


def main(
    startdate,
    enddate,
    xband_task,
    outpath,
    valid_pct_thr=0.05,
    run_radar=True,
    run_lidar=True,
    only_read_data=True,
):
    # Read config
    lidar_cfg = cfg.LIDAR_INFO["vaisala"]
    basepath = cfg.MWSA_DATA_PATH

    lidar_dsize = (120, 70)
    radar_dsize = (360, 866)

    # Define grid for Cartesian mask
    xgrid, ygrid, grid_proj = utils.create_grid(
        cfg.GRID.bbox, cfg.GRID.res, cfg.GRID.res
    )
    grid_proj4 = grid_proj.definition

    get_xband_files = partial(
        file_utils.get_sigmet_file_list_by_task,
        task_name=xband_task,
        # file_regex="([0-9]{12})_VAN.PPI(1_G|2_H|3_H).raw",
    )

    # Util func to get date from xband path
    def xband_date(f):
        return datetime.strptime(os.path.basename(f).split(".")[0], "WRS%y%m%d%H%M%S")

    # Loop over months and get files
    LIDAR_FILES = {}
    XBAND_FILES = {}
    dateinterval = pd.date_range(startdate, enddate, freq="D")
    for day in dateinterval:
        path = os.path.join(basepath, f"{day:%Y/%m/%d}")

        # Get lidar files for the day
        lidar_files = file_utils.find_matching_filenames(
            path,
            lidar_cfg["filepattern"],
            lidar_cfg["timepattern"],
        )
        LIDAR_FILES = {**LIDAR_FILES, **lidar_files}

        # Get xband files for the given task and add to dictionary with time as key
        xband_fn_corr_tasks = get_xband_files(path)
        if len(xband_fn_corr_tasks.keys()) == 0:
            continue

        xband_fn_corr_tasks = xband_fn_corr_tasks[list(xband_fn_corr_tasks.keys())[0]]
        xband_files = {xband_date(f): path + f for f in xband_fn_corr_tasks}

        XBAND_FILES = {**XBAND_FILES, **xband_files}

    LIDAR_LIST = list(LIDAR_FILES.values())
    XBAND_LIST = list(XBAND_FILES.values())

    # Get range, azimuth, elev, lonlatalt for data (assumed to be constant in all scans)
    for f in LIDAR_LIST:
        r = lidar_worker((0, f), return_range_az=True)
        if len(r) == 5:
            lidar_rr = r[1]
            lidar_az = r[2]
            lidar_elev = r[3]
            lidar_lonlatalt = r[4]
            break

    for f in XBAND_LIST:
        r = radar_worker((0, f), return_range_az=True)
        if len(r) == 5:
            xband_rr = r[1]
            xband_az = r[2]
            xband_elev = r[3]
            xband_lonlatalt = r[4]
            break

    logging.info(f"Found {len(LIDAR_LIST)} lidar files!")
    logging.info(f"Found {len(XBAND_LIST)} radar files!")

    if run_lidar:
        # Initialize zarr arrays for storing output values
        lidar_synchronizer = zarr.ProcessSynchronizer(
            str(outpath / f"lidar_{startdate:%Y%m}_{enddate:%Y%m}.sync")
        )
        lidar_data_output = zarr.open_array(
            str(outpath / f"lidar_{startdate:%Y%m}_{enddate:%Y%m}.zarr"),
            mode="w",
            shape=(len(LIDAR_LIST), *lidar_dsize),
            chunks=(1000, *lidar_dsize),
            # dtype='i4',
            synchronizer=lidar_synchronizer,
        )

        # Run calculation as dask bag
        # The output from each worker is an array of same size,
        # so it's handy to stack the results into dask array
        with dask.config.set(
            num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER
        ):
            if not only_read_data:
                bl = db.from_sequence(list(enumerate(LIDAR_LIST)))
                bl.map(lidar_worker, zarr_array=lidar_data_output).compute()
            lidar_arr = da.from_zarr(lidar_data_output)

            n_valid_in_bin = da.count_nonzero(da.isfinite(lidar_arr), axis=0).compute()
            blockage_lidar = (
                np.sum(n_valid_in_bin, axis=1) / lidar_arr.shape[0] / lidar_arr.shape[2]
                < valid_pct_thr
            )

            rr_count_lidar = np.sum(n_valid_in_bin[~blockage_lidar], axis=0)
            n_valid_scans_lidar = (
                ~da.all(da.isnan(lidar_arr), axis=(1, 2)).compute()
            ).sum()
            rr_pct_lidar = (
                rr_count_lidar / n_valid_scans_lidar / (~blockage_lidar).sum()
            )
            pct_lidar = n_valid_in_bin / n_valid_scans_lidar

            lidar_mask = pct_lidar > valid_pct_thr

            del lidar_arr

        ############################################################
        # For lidar data
        # Save pct to file
        hdr = (
            f"Fraction of valid measurements for lidar; "
            f"{n_valid_scans_lidar:.0f} files; "
            f"Elevation {lidar_elev:.2f};"
            f"Created at {datetime.utcnow()} UTC"
        )
        outfn = outpath / f"lidar_obs_pct_{startdate:%Y%m%d}_{enddate:%Y%m%d}.txt"
        save_pct_rr_az(pct_lidar, lidar_rr, lidar_az, hdr, outfn)

        lidar_mask = np.ma.array(data=lidar_mask.astype(float))
        lidar_mask.set_fill_value(np.nan)
        # Grid the mask to Cartesian grid and write to file
        gridded_mask = grid_lidar_mask(
            lidar_mask,
            lidar_rr,
            lidar_az,
            lidar_elev,
            lidar_lonlatalt,
            xgrid,
            ygrid,
            grid_proj4,
            cfg.GRID.rlim,
        )
        outfn = (
            outpath / f"lidar_cart_mask_{startdate:%Y%m%d}_{enddate:%Y%m%d}_"
            f"{cfg.GRID.res:.0f}m_{cfg.GRID.rlim *1e-3:.0f}km.txt"
        )
        save_cart_mask(
            gridded_mask,
            n_valid_scans_lidar,
            "lidar",
            xband_task,
            valid_pct_thr,
            grid_proj4,
            cfg.GRID.res,
            cfg.GRID.rlim,
            outfn,
        )

        df_lidar = pd.Series(data=rr_pct_lidar, index=lidar_rr, name="pct")
        df_lidar.index.name = "range"
        df_lidar.to_csv(
            outpath
            / f"meas_range_lidar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.csv",
        )

        logging.info("Plotting lidar...")
        # Plot percentages
        outfn = (
            outpath
            / f"meas_range_lidar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.png"
        )
        plot_measurement_range(
            rr_pct_lidar,
            lidar_rr,
            n_valid_scans_lidar,
            startdate,
            enddate,
            outfn,
        )

        # Plot binwise percentages
        outfn = (
            outpath
            / f"meas_pct_lidar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.png"
        )
        plot_pct_ppi(pct_lidar, lidar_rr, lidar_az, startdate, enddate, outfn)

    ################################################
    # For radar data
    if run_radar:
        # Itialize zarr arrays for storing output values
        xband_synchronizer = zarr.ProcessSynchronizer(
            str(outpath / f"xband_{startdate:%Y%m}_{enddate:%Y%m}.sync")
        )
        xband_data_output = zarr.open_array(
            str(outpath / f"radar_{startdate:%Y%m}_{enddate:%Y%m}.zarr"),
            mode="w",
            shape=(len(XBAND_LIST), *radar_dsize),
            chunks=(500, *radar_dsize),
            # dtype='i4',
            synchronizer=xband_synchronizer,
        )

        with dask.config.set(
            num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER
        ):
            # Radar data
            if not only_read_data:
                bx = db.from_sequence(list(enumerate(XBAND_LIST)))
                bx.map(radar_worker, zarr_array=xband_data_output).compute()
            xband_arr = da.from_zarr(xband_data_output)
            logging.info("Stacked xband array!")

            n_valid_in_bin = da.sum(da.isfinite(xband_arr), axis=0).compute()
            blockage_xband = (
                np.sum(n_valid_in_bin, axis=1) / xband_arr.shape[0] / xband_arr.shape[2]
                < valid_pct_thr
            )
            logging.info("Calculated blockage!")

            rr_count_xband = np.sum(n_valid_in_bin[~blockage_xband], axis=0)
            logging.info("Calculated count!")
            n_valid_scans_xband = (
                ~da.all(da.isnan(xband_arr), axis=(1, 2)).compute()
            ).sum()
            logging.info("Calculated valid scans!")
            rr_pct_xband = (
                rr_count_xband / n_valid_scans_xband / (~blockage_xband).sum()
            )
            logging.info("Calculated rr_pct!")
            pct_xband = n_valid_in_bin / n_valid_scans_xband
            logging.info("Calculated pct!")

            xband_mask = pct_xband > valid_pct_thr

            del xband_arr

        # Save pct to file
        hdr = (
            f"Fraction of valid measurements for X-band ({xband_task}); "
            f"{n_valid_scans_xband:.0f} files; "
            f"Elevation {xband_elev:.2f};"
            f"Created at {datetime.utcnow()} UTC"
        )
        outfn = outpath / f"xband_obs_pct_{startdate:%Y%m%d}_{enddate:%Y%m%d}.txt"
        save_pct_rr_az(pct_xband, xband_rr, xband_az, hdr, outfn)

        # Grid the mask to Cartesian grid and write to file
        xband_mask = np.ma.array(xband_mask.astype(float))
        xband_mask.set_fill_value(np.nan)
        gridded_mask = grid_radar_mask(
            xband_mask,
            xband_rr,
            xband_az,
            xband_elev,
            xband_lonlatalt,
            xgrid,
            ygrid,
            grid_proj4,
            cfg.GRID.rlim,
        )
        outfn = (
            outpath / f"xband_cart_mask_{startdate:%Y%m%d}_{enddate:%Y%m%d}_"
            f"{cfg.GRID.res:.0f}m_{cfg.GRID.rlim *1e-3:.0f}km.txt"
        )
        save_cart_mask(
            gridded_mask,
            n_valid_scans_xband,
            "xband",
            xband_task,
            valid_pct_thr,
            grid_proj4,
            cfg.GRID.res,
            cfg.GRID.rlim,
            outfn,
        )

        # Save to csv
        df_xband = pd.Series(data=rr_pct_xband, index=xband_rr, name="pct")
        df_xband.index.name = "range"

        df_xband.to_csv(
            outpath
            / f"meas_range_radar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.csv",
        )

        logging.info("Plotting radar...")
        # # Xband
        # Plot percentages
        outfn = (
            outpath
            / f"meas_range_radar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.png"
        )
        plot_measurement_range(
            rr_pct_xband,
            xband_rr,
            n_valid_scans_xband,
            startdate,
            enddate,
            outfn,
        )

        # Plot binwise percentages
        outfn = (
            outpath
            / f"meas_pct_radar_{startdate:%Y%m%d}_{enddate:%Y%m%d}_{xband_task}.png"
        )
        plot_pct_ppi(pct_xband, xband_rr, xband_az, startdate, enddate, outfn)


def grid_lidar_mask(mask, rr, az, elev, lonlatalt, xgrid, ygrid, grid_proj4, rlim):
    """Interpolate a lidar boolean mask to a grid.

    Parameters
    ----------
    mask : np.ma.ndarray
        The mask in polar coordinates.
    rr : np.ndarray
        Range bins in meters.
    az : np.ndarray
        Azimuth angles.
    elev : float
        Elevation angle of the scans.
    lonlatalt : tuple
        Longitude, latitude, altitude of the scans.
    xgrid : np.ndarray
        X-coordinates for grid points.
    ygrid : np.ndarray
        Y-coordinates for grid points.
    grid_proj4 : str
        Grid PROJ4 definition.
    rlim : float
        Distance to which grid is limited.

    Returns
    -------
    np.ma.ndarray
        The Cartesian mask.

    """
    cart, _ = utils.lidar_to_cart(
        mask,
        az,
        rr,
        elev,
        lonlatalt,
        xgrid,
        ygrid,
        grid_proj4=grid_proj4,
        rlim=rlim,
    )
    return cart


def grid_radar_mask(mask, rr, az, elev, lonlatalt, xgrid, ygrid, grid_proj4, rlim):
    """Interpolate a radar boolean mask to a grid.

    Parameters
    ----------
    mask : np.ma.ndarray
        The mask in polar coordinates.
    rr : np.ndarray
        Range bins in meters.
    az : np.ndarray
        Azimuth angles.
    elev : float
        Elevation angle of the scans.
    lonlatalt : tuple
        Longitude, latitude, altitude of the scans.
    xgrid : np.ndarray
        X-coordinates for grid points.
    ygrid : np.ndarray
        Y-coordinates for grid points.
    grid_proj4 : str
        Grid PROJ4 definition.
    rlim : float
        Distance to which grid is limited.

    Returns
    -------
    np.ma.ndarray
        The Cartesian mask.

    """
    cart, _ = utils.radar_to_cart(
        mask,
        az,
        rr,
        elev,
        lonlatalt,
        xgrid,
        ygrid,
        grid_proj4=grid_proj4,
        rlim=rlim,
    )
    return cart


def save_cart_mask(
    mask,
    n_files,
    instrument,
    radar_task,
    pct_thr,
    grid_proj4,
    grid_res,
    grid_rlim,
    outfn,
):
    """Save mask of Cartesian fraction of available measurements.

    Parameters
    ----------
    mask : np.ma.ndarray
        The mask of fraction of available measurements in Cartesian grid.
    n_files : int
        Number of scans used to calculate.
    instrument : str
        Instrument name.
    radar_task : str
        Radar task name.
    pct_thr : float
        Threshold value used to calculate mask.
    grid_proj4 : str
        PROJ4 string for the grid.
    grid_res : float
        Grid resolution
    grid_rlim : float
        Maximum distance of grid.
    outfn : str
        Output file path.

    """
    # Save mask to file
    # Basic documentation in header
    hdr = (
        f"Observation mask for {instrument} ({radar_task}); "
        f"{n_files:.0f} files, thr={pct_thr} of valid measurements; "
        f"Grid proj: {grid_proj4}, resolution {grid_res}x{grid_res}m, "
        f"limit {grid_rlim}m;\n"
        f"Created at {datetime.utcnow()} UTC"
    )
    mask = mask.astype(int)
    mask.set_fill_value(0)
    np.savetxt(outfn, mask.filled(), fmt="%1.1d", header=hdr)


def save_pct_rr_az(pct, rr, az, header, outfn):
    """Save results in txt files.

    Parameters
    ----------
    pct : np.ndarray
        Fraction available measurements for each range
    rr : np.ndarray
        Range bins in meters.
    az : np.ndarray
        Azimuth angles.
    header : str
        Header that is saved to txt files.
    outfn : pathlib.Path
        Output file path. A suffix is added to the path to indicate measurement
        fractions, range, and azimuth data files.

    """
    base_fn = outfn.stem
    # Data
    data_fn = outfn.with_name(f"{base_fn}_pct.txt")
    np.savetxt(data_fn, pct, header=header)

    # Range
    rr_fn = outfn.with_name(f"{base_fn}_range.txt")
    np.savetxt(rr_fn, rr, header=header)

    # Azimuth
    az_fn = outfn.with_name(f"{base_fn}_azimuth.txt")
    np.savetxt(az_fn, az, header=header)


def plot_measurement_range(rr_pct, rr, n_valid_scans, startdate, enddate, outfn):
    """Plot fraction of available measurements as function of range.

    Parameters
    ----------
    rr_pct : np.ndarray
        Fraction available measurements for each range
    rr : np.ndarray
        Range bins in meters.
    n_valid_scans : int
        Number of scans used to calculate `rr_pct`, written in image.
    startdate : datetime.datetime
        Starting date of data, written in image.
    enddate : datetime.datetime
        Ending date of data, written in image.
    outfn : str
        Output filename.

    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        rr * 1e-3,
        rr_pct,
        "b",
        label=f"Number of valid lidar scans: {n_valid_scans}",
        lw=2,
    )

    ax.set_xlim([0, 15])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.legend()
    ax.grid(which="both")

    ax.set_ylabel("Percentage")
    ax.set_xlabel("Range [km]")

    ax.set_title(
        f"Percentage of available measurements\n"
        f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}"
    )
    fig.savefig(outfn, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_pct_ppi(pct, rr, az, startdate, enddate, outfn):
    """Plot a PPI image of the valid measurement fractions.

    Parameters
    ----------
    pct : (N,M) np.ma.array
        Array (azimuth, range) that is plotted as PPI.
    rr : (M,) np.array
        Range bins in meters.
    az : (N,) np.array
        Azimuth angles.
    startdate : datetime.datetime
        Starting date of data, written in image.
    enddate : datetime.datetime
        Ending date of data, written in image.
    outfn : str
        Output filename.

    """
    fmt = mlt.ticker.StrMethodFormatter("{x:.0f}")
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    p = plotting.plot_ppi(
        ax,
        pct,
        az,
        rr * 1e-3,
        rasterized=True,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)
    cbar = plt.colorbar(p, orientation="vertical", cax=cax, ax=None)
    cbar.set_label("Percentage", weight="bold")
    cbar.ax.tick_params(labelsize=12)

    # x-axis
    ax.set_xlabel("Distance from site (km)")
    ax.set_title(ax.get_title(), y=-0.22)
    ax.xaxis.set_major_formatter(fmt)

    # y-axis
    ax.set_ylabel("Distance from site (km)")
    ax.yaxis.set_major_formatter(fmt)

    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect(1)
    ax.grid(zorder=0, linestyle="-", linewidth=0.4)
    ax.set_title(
        f"Percentage of available measurements\n"
        f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}"
    )
    fig.savefig(outfn, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "startdate", type=str, help="the startdate (YYYYmmdd) (only month considered"
    )
    argparser.add_argument(
        "enddate", type=str, help="the enddate (YYYYmmdd) (only month considered"
    )
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

    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m%d")
    enddate = datetime.strptime(args.enddate, "%Y%m%d")

    outpath = Path(args.outpath)

    # Set style file
    plt.style.use(cfg.STYLE_FILE)

    logging.basicConfig(level=logging.INFO)

    main(
        startdate,
        enddate,
        args.task_name,
        outpath,
        run_radar=True,
        run_lidar=True,
        only_read_data=args.only_read,
    )
