"""Grid lidar and X band radar measurements.

Grids measurements and calculates e.g. number of available measurements for each
pair of scans.

Configurations in config.py

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>
"""
import os
import argparse
import warnings
import logging
from functools import partial
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from sklearn import metrics
from scipy.ndimage.filters import generic_filter
import dask
import zarr
from netCDF4 import Dataset
import pyart
from pyart.io.sigmet import SigmetFile

import utils
import file_utils
import config as cfg
import vcor_dual_prf.vcor_dual_prf.vcor_dual_prf as vcor

warnings.simplefilter(action="ignore")


# Create grid globally so its available in worker
minmax_vel = 30
xgrid, ygrid, grid_proj = utils.create_grid(cfg.GRID.bbox, cfg.GRID.res, cfg.GRID.res)
grid_proj4 = grid_proj.definition


def worker(
    lidar_time,
    lidar_fn,
    xband_time,
    xband_fn,
    xl_zarr=None,
):
    """Get gridded data for the lidar-radar comparison.

    Parameters
    ----------
    lidar_time : datetime.datetime
        Time of lidar file.
    lidar_fn : str
        Lidar filepath.
    xband_time : datetime.datetime
        Time of X-band file.
    xband_fn : str
        X-band filepath.
    xl_zarr : zarr array
        zarr array where gridded measurements are saved.

    Returns
    -------
    dict
        Dictionary of linear fit results and number of available measurements.

    """
    res_dict = {}
    # First get data
    try:
        lidar = Dataset(lidar_fn, "r", format="NETCDF4")
    except (ValueError, OSError, IOError):
        logging.error(f"Failed to read {lidar_fn}")
        return
    try:
        xband = pyart.io.read_sigmet(xband_fn)
        vcor.correct_dualprf(
            radar=xband,
            two_step=True,
            method_det="median",
            kernel_det=np.ones((3, 3)),
            vel_field="velocity",
            new_field="velocity",
            replace=True,
        )
    except (ValueError, OSError, IOError, IndexError):
        logging.error(f"Failed to read {xband_fn}")
        return

    dlidar = utils.get_windcube_lidar_data(
        lidar,
        use_quality_flag=True,
        CNR_thr=cfg.CNR_THR,
        QIND_thr=0,
    )
    dxband = utils.get_radar_data(xband, alt=35, SQI_thr=None)

    # These are no longer used
    del lidar, xband

    # Estimate Xband Doppler variance
    # According to Eq. 6.23 in Doviak&Zrnic 2006
    f = SigmetFile(xband_fn)
    # In meters
    wavelength = (
        f.ingest_header["task_configuration"]["task_misc_info"]["wavelength"] * 1e-4
    )
    M = f.ingest_header["task_configuration"]["task_dsp_info"]["sample_size"]
    Ts = 1 / f.ingest_header["task_configuration"]["task_dsp_info"]["prf"]
    xband_var_v = dxband["WRAD"] * wavelength / (8 * M * Ts * np.sqrt(np.pi))

    # Estimate SNR (for formula, see email "SNR:n käyttö luokittelussa" from Jarmo)
    # Eq: SNR = 10 \log ((S+Nm)-N)/N) = dBZ  - dBZ0 - 20\log(r/1 km)
    # where dBZ0 is stored as the reflectivity calibration constant
    # NEZ_xband = (SigmetFile(xband_fn).ingest_header["task_configuration"]
    #              ["task_calib_info"]["reflectivity_calibration"] / 16)
    # R_xband = np.tile(dxband["r_los"], (dxband.nrays, 1)) * 1e-3
    # SNR_xband = dxband["reflectivity"] - NEZ_xband - 20 * np.log10(R_xband)

    dlidar["lonlatalt"] = (
        dlidar["lonlat"][0],
        dlidar["lonlat"][1],
        cfg.LIDAR_INFO["vaisala"]["altitude"],
    )
    dxband["lonlatalt"] = (
        dxband["lonlat"][0],
        dxband["lonlat"][1],
        cfg.RADAR_INFO["fivxt"]["altitude"],
    )

    if any(np.isnan(dlidar["lonlatalt"])):
        # Lidar location can be nan in some rare cases; so use xband location instead
        dlidar["lonlatalt"] = dxband["lonlatalt"]

    # Grid the data
    cart_lidar, _ = utils.lidar_to_cart(
        dlidar["V"],
        dlidar["azimuth"],
        dlidar["r_los"],
        dlidar["elev"],
        dlidar["lonlatalt"],
        xgrid,
        ygrid,
        grid_proj4=grid_proj4,
        rlim=cfg.GRID.rlim,
    )
    cart_lidar_cnr, _ = utils.lidar_to_cart(
        np.ma.array(data=dlidar["cnr"], mask=dlidar["V"].mask),
        dlidar["azimuth"],
        dlidar["r_los"],
        dlidar["elev"],
        dlidar["lonlatalt"],
        xgrid,
        ygrid,
        grid_proj4=grid_proj4,
        rlim=cfg.GRID.rlim,
    )
    cart_xband, _ = utils.radar_to_cart(
        dxband["V"],
        dxband["azimuth"],
        dxband["r_los"],
        dxband["elev"],
        dxband["lonlatalt"],
        xgrid,
        ygrid,
        grid_proj4=grid_proj4,
        rlim=cfg.GRID.rlim,
    )
    cart_xband_median = generic_filter(
        np.abs(cart_xband), np.nanmedian, size=cfg.RADAR_MEDIAN_FILTER_WINDOW
    )
    cart_xband.mask[
        np.abs(cart_xband) > cfg.RADAR_MEDIAN_FILTER_FACTOR * cart_xband_median
    ] = True

    try:
        cart_xband_var_v, _ = utils.radar_to_cart(
            xband_var_v,
            dxband["azimuth"],
            dxband["r_los"],
            dxband["elev"],
            dxband["lonlatalt"],
            xgrid,
            ygrid,
            grid_proj4=grid_proj4,
            rlim=(cfg.GRID.rlim),
        )
        cart_xband_snr, _ = utils.radar_to_cart(
            dxband["SNR"],
            dxband["azimuth"],
            dxband["r_los"],
            dxband["elev"],
            dxband["lonlatalt"],
            xgrid,
            ygrid,
            grid_proj4=grid_proj4,
            rlim=(cfg.GRID.rlim),
        )
        cart_xband_reflectivity, _ = utils.radar_to_cart(
            dxband["reflectivity"],
            dxband["azimuth"],
            dxband["r_los"],
            dxband["elev"],
            dxband["lonlatalt"],
            xgrid,
            ygrid,
            grid_proj4=grid_proj4,
            rlim=(cfg.GRID.rlim),
        )
    except ValueError:
        return None

    # Read mask
    mask = np.loadtxt(cfg.OBS_MASK_PATH).astype(bool)
    lidar_nobs_mask = cart_lidar.mask | (~mask)
    xband_nobs_mask = cart_xband.mask | (~mask)

    count_lidar = np.sum(~lidar_nobs_mask)
    count_xband = np.sum(~xband_nobs_mask)
    count_union = np.sum((~lidar_nobs_mask) | (~xband_nobs_mask))
    count_unobserved = np.sum((lidar_nobs_mask) & (xband_nobs_mask))
    count_common = np.sum((~lidar_nobs_mask) & (~xband_nobs_mask))

    res_dict["nobs_lidar"] = count_lidar
    res_dict["nobs_xband"] = count_xband
    res_dict["nobs_union"] = count_union
    res_dict["nobs_unobserved"] = count_unobserved
    res_dict["nobs_common"] = count_common

    # Fraction of valid bins
    lidar_blockage_mask = np.loadtxt(cfg.POLAR_OBS_MASK_LIDAR_PATH).astype(float)
    xband_blockage_mask = np.loadtxt(cfg.POLAR_OBS_MASK_XBAND_PATH).astype(float)

    res_dict["frac_valid_bins_lidar"] = (
        np.sum(~dlidar["V"].mask)
        / (lidar_blockage_mask >= cfg.POLAR_OBS_MASK_THR).sum()
    )

    xr_mask = dxband["r_los"] <= dlidar["r_los"].max()
    res_dict["frac_valid_bins_xband"] = (
        np.sum(~dxband["V"].mask[:, xr_mask])
        / (xband_blockage_mask[:, xr_mask] >= cfg.POLAR_OBS_MASK_THR).sum()
    )

    n_min_angles = 9
    rlidar_mask = np.sum(~dlidar["V"].mask, axis=0) >= (n_min_angles / 3)
    if np.any(rlidar_mask):
        res_dict["max_valid_range_lidar"] = np.max(dlidar["r_los"][rlidar_mask])
    else:
        res_dict["max_valid_range_lidar"] = 0

    rxband_mask = np.sum(~dxband["V"].mask[:, xr_mask], axis=0) >= (n_min_angles / 1)
    if np.any(rxband_mask):
        res_dict["max_valid_range_xband"] = np.max(
            dxband["r_los"][xr_mask][rxband_mask]
        )
    else:
        res_dict["max_valid_range_xband"] = 0

    # Calculate rain rate
    Z = dxband["reflectivity"][:, dxband["r_los"] < cfg.GRID.rlim]
    R = np.ma.power(np.ma.power(10.0, Z / 10.0) / 223, 1 / 1.53)
    res_dict["rr_max"] = np.ma.max(R)
    res_dict["rr_mean"] = np.ma.mean(R)
    res_dict["rr_median"] = np.ma.median(R)
    res_dict["rr_Q95"] = np.nanquantile(R.data, 0.95)

    try:
        # Calculate descriptive statistics
        # FOR LIDAR AND XBAND
        xl_mask, xl_stats, xl_ols_results, xl_rlm_results = calculate_comparison_stats(
            cart_lidar, cart_xband
        )

        # Combine to dicts
        fit_dict = linear_fit_to_dict(
            xl_ols_results,
            xl_rlm_results,
        )
        stats_dict = stats_to_dict(xl_stats)

    except Exception:
        stats_dict = {}
        fit_dict = {}

    rdict = {**stats_dict, **fit_dict, **res_dict}
    rdict["lidar_time"] = lidar_time
    rdict["xband_time"] = xband_time
    rdict["n_lidar"] = np.isfinite(cart_lidar.filled()).sum()
    rdict["n_xband"] = np.isfinite(cart_xband.filled()).sum()

    if len(stats_dict.keys()):
        n_orig = cart_xband.filled().ravel().size
        xl_arr = np.stack(
            [
                np.repeat([xband_time.timestamp()], n_orig),
                np.repeat([lidar_time.timestamp()], n_orig),
                cart_xband.filled().ravel(),
                cart_lidar.filled().ravel(),
                cart_lidar_cnr.filled().ravel(),
                cart_xband_var_v.filled().ravel(),
                cart_xband_snr.filled().ravel(),
                cart_xband_reflectivity.filled().ravel(),
            ],
            axis=-1,
        )[xl_mask.ravel(), :]
        xl_zarr.append(xl_arr)

        rdict["n_xband_lidar"] = np.sum(xl_mask)
        del xl_arr, xl_mask
    # Remove variables to release memory "more aggressively"
    # (since everything is inside a function, the memory would be removed eventually,
    # but for multiprocessing we want to release it faster)
    del cart_lidar, cart_xband, cart_xband_median
    del mask, lidar_nobs_mask, xband_nobs_mask
    del cart_lidar_cnr, cart_xband_var_v, cart_xband_snr, cart_xband_reflectivity
    del dlidar, dxband

    return rdict


def linear_fit_to_dict(
    xl_ols_res,
    xl_rlm_res,
):
    d = {}
    d["xl_OLS_intercept"] = xl_ols_res.params[0]
    d["xl_OLS_slope"] = xl_ols_res.params[1]
    d["xl_OLS_intercept_ci_0.025"] = xl_ols_res.conf_int(0.05)[0, 0]
    d["xl_OLS_intercept_ci_0.975"] = xl_ols_res.conf_int(0.05)[0, 1]
    d["xl_OLS_slope_ci_0.025"] = xl_ols_res.conf_int(0.05)[1, 0]
    d["xl_OLS_slope_ci_0.975"] = xl_ols_res.conf_int(0.05)[1, 1]
    d["xl_OLS_intercept_stderr"] = xl_ols_res.bse[0]
    d["xl_OLS_slope_stderr"] = xl_ols_res.bse[1]

    d["xl_RLM_intercept"] = xl_rlm_res.params[0]
    d["xl_RLM_slope"] = xl_rlm_res.params[1]
    d["xl_RLM_intercept_ci_0.025"] = xl_rlm_res.conf_int(0.05)[0, 0]
    d["xl_RLM_intercept_ci_0.975"] = xl_rlm_res.conf_int(0.05)[0, 1]
    d["xl_RLM_slope_ci_0.025"] = xl_rlm_res.conf_int(0.05)[1, 0]
    d["xl_RLM_slope_ci_0.975"] = xl_rlm_res.conf_int(0.05)[1, 1]
    d["xl_RLM_intercept_stderr"] = xl_rlm_res.bse[0]
    d["xl_RLM_slope_stderr"] = xl_rlm_res.bse[1]

    return d


def stats_to_dict(xl_stats):
    dxl = DescrStatsW2data(xl_stats)
    dxl = {f"xl_{key}": val for key, val in dxl.items()}

    return dxl


def calculate_comparison_stats(data1, data2):
    mask = (~data1.mask) & (~data2.mask)
    data1_masked = data1[mask].filled().ravel()
    data2_masked = data2[mask].filled().ravel()
    data = np.stack(
        [
            data2_masked,
            data1_masked,
        ]
    ).T

    stats = DescrStatsW(data, weights=None)
    stats.rmse = metrics.mean_squared_error(data1_masked, data2_masked, squared=False)
    stats.nobs_var = np.array(
        [
            (~data2.mask).sum(),
            (~data1.mask).sum(),
        ]
    )
    stats.min_var = np.nanmin(data, axis=0)
    stats.max_var = np.nanmax(data, axis=0)

    # linear regression
    X = sm.add_constant(data1_masked)

    ols_model = sm.OLS(data2_masked, X)
    ols_results = ols_model.fit()

    # Robust linear regression
    rlm_model = sm.RLM(data2_masked, X)
    rlm_results = rlm_model.fit()

    # Clear variables
    del data, data1_masked, data2_masked

    # Return
    return mask, stats, ols_results, rlm_results


def DescrStatsW2data(s):
    return {
        "mean_1": s.mean[0],
        "mean_2": s.mean[1],
        "std_var1": s.std[0],
        "std_var2": s.std[1],
        "std_mean_var1": s.std_mean[0],
        "std_mean_var2": s.std_mean[1],
        "corrcoef": s.corrcoef[0, 1],
        "nobs": s.nobs,
        "nobs_var1": s.nobs_var[0],
        "nobs_var2": s.nobs_var[1],
        "min_var1": s.min_var[0],
        "min_var2": s.min_var[1],
        "max_var1": s.max_var[0],
        "max_var2": s.max_var[1],
        "rmse": s.rmse,
    }


# @profile
def main(startdate, enddate, radar_type, outpath):
    # Read config
    lidar_cfg = cfg.LIDAR_INFO["vaisala"]
    basepath = cfg.MWSA_DATA_PATH

    get_xband_files = partial(
        file_utils.get_sigmet_file_list_by_task, task_name=radar_type
    )

    # Itialize zarr arrays for storing output values
    xl_synchronizer = zarr.ProcessSynchronizer(outpath / f"xl_{radar_type}.sync")
    xl_data_output = zarr.open_array(
        str(outpath / f"scatterplot_{startdate:%Y%m}_{enddate:%Y%m}_{radar_type}_xl"),
        mode="w",
        shape=(1, 8),
        chunks=(1000, 8),
        # dtype='i4',
        synchronizer=xl_synchronizer,
    )

    kw_args = {
        "xl_zarr": xl_data_output,
    }

    # Util func to get date from xband path
    xband_date = lambda f: datetime.strptime(
        os.path.basename(f).split(".")[0], "WRS%y%m%d%H%M%S"
    )
    # Loop over months and get files
    res = []
    dateinterval = pd.date_range(startdate, enddate + pd.offsets.MonthEnd(), freq="M")
    for month in dateinterval:
        path = os.path.join(basepath, f"{month:%Y/%m/}")

        # Get lidar files for the month
        lidar_files = file_utils.find_matching_filenames(
            path,
            lidar_cfg["filepattern"],
            lidar_cfg["timepattern"],
        )
        # Get xband files for the given task and add to dictionary with time as key
        xband_fn_corr_tasks = get_xband_files(path)
        xband_fn_corr_tasks = xband_fn_corr_tasks[list(xband_fn_corr_tasks.keys())[0]]
        xband_files = {xband_date(f): path + f for f in xband_fn_corr_tasks}

        # For each lidar file, get closest xband and cband file
        for lidar_time, lidar_fn in sorted(lidar_files.items()):
            xband_time, xband_fn, td = file_utils.find_closest_file(
                lidar_time, xband_files
            )

            if td > 120:
                continue

            res.append(
                dask.delayed(worker)(
                    lidar_time,
                    lidar_fn,
                    xband_time,
                    xband_fn,
                    **kw_args,
                )
            )
    res = dask.compute(
        *res, num_workers=cfg.DASK_NWORKERS, scheduler=cfg.DASK_SCHEDULER
    )

    # Gather results
    res_dicts = [r for r in res if r is not None]
    # Dictionaries into dataframe
    df = pd.DataFrame(res_dicts)
    df.set_index("lidar_time", inplace=True)

    # Calculate time differences

    def f(t1, t2):
        return (t1 - t2).seconds if t1 > t2 else (-1) * (t2 - t1).seconds

    df["time_diff_lidar_xband"] = [
        f(t1, t2)
        for t1, t2 in zip(
            [t.to_pydatetime() for t in df["xband_time"]],
            [t.to_pydatetime() for t in df.index],
        )
    ]

    # Dump as csv
    fn = f"{startdate:%Y%m}_{enddate:%Y%m}_{radar_type}_stats.csv"
    df.to_csv(outpath / fn, index_label="lidar_time")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "radar_type",
        type=str,
        help="X-band type",
        choices=["WND-01", "WND-02", "WND-03", "MWS-PPI1_G"],
    )
    argparser.add_argument("startdate", type=str, help="the startdate (YYYYmm)")
    argparser.add_argument("enddate", type=str, help="the enddate (YYYYmm)")
    argparser.add_argument("--outpath", type=str, default=".", help="Output path")

    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m")
    enddate = datetime.strptime(args.enddate, "%Y%m")

    outpath = Path(args.outpath)
    outpath.mkdir(exist_ok=True, parents=True)

    main(startdate, enddate, args.radar_type, outpath)
