"""Plot measurement availability distributions.

Requires csv-output from compute_gridded_lidar_xband.py

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
import re
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import warnings

warnings.simplefilter(action="ignore")
import locale

from radar_plotting.plotting import final
import config as cfg
import utils

round_to_multiple = lambda number, multiple: multiple * (1 + (number - 1) // multiple)

# FuncFormatter can be used as a decorator
@ticker.FuncFormatter
def m2km_formatter(x, pos):
    return f"{x / 1000:.0f}"


@ticker.FuncFormatter
def m_formatter(x, pos):
    return f"{x:.0f}"


@ticker.FuncFormatter
def mmh_formatter(x, pos):
    return f"{x:.1f}"


def plot_regression_timeseries_with_weather(
    fig, df, path, prefix="xl", roll_window="30min", datesuffix="", outpath=Path(".")
):
    """Plot a timeseries of linear regression coefficients with surface observations.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure object.
    df : pandas.DataFrame
        The dataframe with each row representing one paired lidar & radar scan.
    path : str
        refix added to filename.
    prefix : str, optional
        Variable prefix, by default "xl"
    roll_window : str, optional
        Window size for rolling mean, by default "30min"
    datesuffix : str, optional
        Date suffix added to the end of filename, by default ""
    outpath : _type_, optional
        Output directory, by default Path(".")

    """
    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    # Plot a simple timeseries of statistics
    times = df.index.to_pydatetime()
    timediffs = np.diff(times)

    datelims = []
    start = times[0]
    for i in np.where(timediffs > timedelta(minutes=600))[0]:
        datelims.append((start, times[i]))
        start = times[i + 1]
    datelims.append((start, times[-1]))

    n_periods = len(datelims)
    widths = [v[0] for v in np.diff(datelims) / np.max(np.diff(datelims))]

    alpha = 0.6
    col1 = cfg.COLORS.C1
    col2 = cfg.COLORS.C2
    col0 = cfg.COLORS.C0

    axes = fig.subplots(
        nrows=4,
        ncols=n_periods,
        sharey="row",
        sharex="col",
        gridspec_kw=dict(width_ratios=widths, height_ratios=[1, 1, 1, 1]),
        squeeze=False,
    )

    wvar1 = "CLHB_PT1M_INSTANT"
    wvar2 = "WS_PT10M_AVG"
    wvar3 = "rr_mean"

    wvar2_axes = []
    wvar1_axes = []

    wvar3_lims = (0, 5)
    wvar2_lims = (
        df["WS_PT10M_AVG"].min(),
        df["WS_PT10M_AVG"].max(),
    )
    wvar1_lims = (
        np.floor(df["CLHB_PT1M_INSTANT"].min() / 100) * 100,
        np.ceil(df["CLHB_PT1M_INSTANT"].max() / 100) * 100,
    )

    wax = ax1 = ax2 = ax3 = None
    for i, (start, end) in enumerate(datelims):
        tdf = df.loc[start:end]
        # import ipdb; ipdb.set_trace()
        tdf.dropna(axis=0, subset=["radar_type"], inplace=True)

        wax = axes[-1, i]
        ax1 = axes[0, i]
        ax2 = axes[2, i]
        ax3 = axes[1, i]

        # Regression slope
        ax1.plot_date(
            tdf.index,
            tdf[f"{prefix}_OLS_slope"],
            color=col1,
            marker=",",
            markersize=0.05,
            alpha=alpha,
            ls="--",
            label=f"OLS slope",
        )
        ax1.plot_date(
            tdf.index,
            tdf[f"{prefix}_OLS_slope"].rolling(roll_window).mean(),
            color=col1,
            ls="-",
            lw=0.5,
            label=f"OLS slope",
            marker=None,
        )

        ax1.plot_date(
            tdf.index,
            tdf[f"{prefix}_RLM_slope"],
            color=col2,
            marker=",",
            markersize=0.05,
            alpha=alpha,
            ls="--",
            label=f"RLM slope",
        )
        ax1.plot_date(
            tdf.index,
            tdf[f"{prefix}_RLM_slope"].rolling(roll_window).mean(),
            color=col2,
            ls="-",
            lw=0.5,
            label=f"RLM slope",
            marker=None,
        )
        ax1.set_ylim([-0.5, 1.5])
        # axhline takes x coordinates in axis coordinates, not date
        ax1.axhline(y=1, xmin=0, xmax=1, ls="--", color="k")

        # Number of observations
        ax3.plot_date(
            tdf.index,
            tdf[f"frac_valid_bins_lidar"],
            color=col1,
            ls="-",
            marker=None,
            markersize=0.05,
            alpha=alpha,
        )
        ax3.plot_date(
            tdf.index,
            tdf[f"frac_valid_bins_xband"],
            color=col2,
            ls="-",
            marker=None,
            markersize=0.05,
            alpha=alpha,
        )
        ax3.set_ylim([0, 1])

        # Number of observations
        ax2.plot_date(
            tdf.index,
            tdf[f"n_union_normalized"],
            color=col0,
            ls="-",
            marker=None,
            markersize=0.05,
            alpha=alpha,
        )
        ax2.plot_date(
            tdf.index,
            tdf[f"n_common_normalized"],
            color="tab:green",
            ls="-",
            marker=None,
            markersize=0.05,
            alpha=alpha,
        )
        ax2.set_ylim([0, 1])

        # Weather data
        # Multiple axis
        par1 = wax.twinx()
        par2 = wax.twinx()
        if len(wvar2_axes) > 0:
            par2.get_shared_y_axes().join(*wvar2_axes)
        if len(wvar1_axes) > 0:
            par1.get_shared_y_axes().join(*wvar1_axes)

        wvar1_axes.append(par1)
        wvar2_axes.append(par2)
        # no x-ticks
        par2.xaxis.set_ticks([])

        (temp,) = wax.plot_date(
            tdf.index, tdf[wvar1], fmt="-", color="tab:red", lw=0.5, zorder=10
        )

        (ws,) = par1.plot_date(
            tdf.index, tdf[wvar2], fmt="-", color="tab:blue", lw=0.5, zorder=0
        )

        (vis,) = par2.plot_date(
            tdf.index, tdf[wvar3], fmt="-", color="k", lw=0.5, zorder=10
        )

        par1.set_ylim(wvar2_lims)
        par2.set_ylim(wvar3_lims)
        wax.set_ylim(wvar1_lims)

        for ax, p in zip([wax, par1, par2], [temp, ws, vis]):
            ax.set_yticklabels([])
            ax.yaxis.label.set_color(p.get_color())
            ax.tick_params(axis="y", colors=p.get_color())

        for ax in [ax1, ax2, ax3, wax]:
            ax.grid()
            ax.set_xticklabels([])
            # ax.legend(loc=1, ncol=2)

        hours = mdates.HourLocator(
            byhour=[
                0,
                6,
                12,
                18,
            ],
            interval=1,
        )
        locator = mdates.AutoDateLocator(
            minticks=5, maxticks=30, interval_multiples=True
        )
        # Set daily intervals in the locator
        locator.intervald[3] = [1, 2]
        formatter = mdates.ConciseDateFormatter(locator, show_offset=True)
        wax.set_xlim((tdf.index.min(), tdf.index.max()))
        wax.xaxis.set_major_locator(locator)
        wax.xaxis.set_major_formatter(formatter)
        wax.xaxis.set_minor_locator(hours)
        plt.setp(wax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set yaxis labels and limits
    axes[1, 0].set_ylabel("N (B-scan)")
    axes[0, 0].set_ylabel("Slope")
    axes[2, 0].set_ylabel("N")

    # Boxes for "legend"
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(
        0.94,
        0.8,
        "OLS",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color=col1,
    )
    plt.text(
        0.94,
        0.75,
        "RLM",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color=col2,
    )

    plt.text(
        0.94,
        0.53,
        "Lidar",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color=col1,
    )
    plt.text(
        0.94,
        0.50,
        "X-band",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color=col2,
    )
    plt.text(
        0.94,
        0.47,
        "Union",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color="k",
    )
    plt.text(
        0.94,
        0.44,
        "Intersection",
        transform=fig.transFigure,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
        color="tab:green",
    )

    # Set yaxis in last columns
    offset = 40
    # Set extra y axis spines
    # right, left, top, bottom
    # vis_ax = wvar2_axes[-1].get_shared_yaxis().join(*wvar2_axes)
    par2.spines["right"].set_position(("outward", offset))
    par2.set_ylabel("Mean rain rate [mm h$^{-1}$]")
    par2.set_ylim(wvar3_lims)
    par2.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: f"{x:.1f}")
    )

    axes[-1, 0].set_ylabel("Cloud base height [100m]")
    axes[-1, 0].set_ylim(wvar1_lims)
    axes[-1, 0].get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x / 1e2), ","))
    )

    par1.set_ylabel("Wind speed [m/s]")
    par1.set_ylim(wvar2_lims)
    par1.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: f"{x:.1f}")
    )

    fig.subplots_adjust(wspace=0.1)
    fig.suptitle(
        (
            f"{df.index.min():%Y/%m/%d %H:%M} - "
            f"{df.index.max():%Y/%m/%d %H:%M} UTC "
            f"{df['radar_type'].unique()[0]}"
        ),
        y=0.95,
    )
    fig.savefig(
        outpath / f"{path}_{df['radar_type'].unique()[0]}"
        f"_{prefix}_regression_timeseries_weather_{datesuffix}.png"
    )
    plt.close()


def plot_nobs_distribution(
    fig,
    df,
    path,
    x="VIS_PT1M_AVG",
    xmin=-10,
    xmax=10,
    ny=20,
    nmin=10,
    vmin=10,
    vmax=1e3,
    x_formatter=m2km_formatter,
    x_unit="km",
    xres=1,
    datesuffix="",
    outpath=Path("."),
    title=True,
):
    """Plot distribution of fraction of available measurements against surface measurement.

    Parameters
    ----------
    fig : matplotlib.figure
        Figure object
    df : pandas.DataFrame
        The dataframe with each row representing one paired lidar & radar scan.
    path : str
        Prefix added to filename.
    x : str, optional
        Surface station measurement name, by default "VIS_PT1M_AVG"
    xmin : int, optional
        Lower limit for surface station measurement values, by default -10
    xmax : int, optional
        Upper limit for surface station measurement values, by default 10
    ny : int, optional
        Number of bins in plt.hexbin for fraction of available measurements, by default 20
    nmin : int, optional
        Minimum count of points in bin to plt.hexbin, by default 10
    vmin : int, optional
        Minimum value for colorbar norm, by default 10
    vmax : _type_, optional
        Maximum value for colorbar norm, by default 1e3
    x_formatter : callable, optional
        Matplotlib FuncFormatter to format surface measurement tick labels, by default m2km_formatter
    x_unit : str, optional
        Unit for surface station measurements, by default "km"
    xres : int, optional
        Resolution of surface station measurements, used to calculate bin size for plt.hexbin,
        by default 1
    datesuffix : str, optional
        Date suffix added to the end of filename, by default ""
    outpath : pathlib.Path, optional
        Output directory, by default Path(".")
    title : bool, optional
        Whether to write a title to plot, by default True

    """
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, height_ratios=[1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)

    dfs = df.melt(
        id_vars=["utctime", x],
        value_vars=["frac_valid_bins_lidar", "frac_valid_bins_xband"],
        var_name="instrument",
        value_name="frac_valid",
    )
    dfs.set_index("utctime", inplace=True)

    dfs["instrument"][dfs["instrument"] == "frac_valid_bins_lidar"] = "Lidar"
    dfs["instrument"][dfs["instrument"] == "frac_valid_bins_xband"] = "X-band"

    # import ipdb; ipdb.set_trace()
    N_lidar = dfs[dfs["instrument"] == "Lidar"]["frac_valid"].notna().sum()
    N_xband = dfs[dfs["instrument"] == "X-band"]["frac_valid"].notna().sum()

    hue_norm = mpl.colors.LogNorm(vmin=vmin * 100, vmax=vmax * 100)
    p = ax1.hexbin(
        dfs[dfs["instrument"] == "Lidar"][x],
        dfs[dfs["instrument"] == "Lidar"]["frac_valid"],
        C=np.ones_like(dfs[dfs["instrument"] == "Lidar"]["frac_valid"], dtype=np.float)
        * 100
        / N_lidar,
        bins="log",
        norm=hue_norm,
        gridsize=(int((xmax - xmin) / xres), ny),
        mincnt=nmin,
        reduce_C_function=np.sum,
    )
    p = ax2.hexbin(
        dfs[dfs["instrument"] == "X-band"][x],
        dfs[dfs["instrument"] == "X-band"]["frac_valid"],
        C=np.ones_like(dfs[dfs["instrument"] == "X-band"]["frac_valid"], dtype=np.float)
        * 100
        / N_xband,
        bins="log",
        norm=hue_norm,
        gridsize=(int((xmax - xmin) / xres), ny),
        mincnt=nmin,
        reduce_C_function=np.sum,
    )
    cbar = plt.colorbar(
        p,
        label="% of cases",
        ax=[ax1, ax2],
        aspect=50,
        orientation="horizontal",
        shrink=0.6,
        extend="max",
        location="top",
        format=ticker.StrMethodFormatter("{x:.1f}"),
    )
    # cbar.axes.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # FuncFormatter can be used as a decorator
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x / 1000:.0f}"

    pick_outside_brackets = re.compile("([^[\]]+)(?:$|(?=\[))")
    xname = pick_outside_brackets.findall(names[x])[0]
    title_y = -0.25
    for ax, inst in zip([ax1, ax2], ["(a) Lidar", "(b) X-band radar"]):
        ax.set_xlabel(f"{xname}[{x_unit}]")
        # ax.set_xlabel(f"{names[x]}")
        ax.set_title(inst, y=title_y)
        ax.set_xlim((xmin, xmax))
        ax.xaxis.set_major_formatter(x_formatter)
    ax1.set_ylim((0, 1))
    ax1.set_ylabel("Fraction of valid measurements")

    if title:
        fig.suptitle(
            f"{df.index.min():%Y/%m/%d %H:%M} - "
            f"{df.index.max():%Y/%m/%d %H:%M} UTC\n{df['radar_type'].unique()[0]}"
        )
    plt.savefig(
        outpath / f"{path}_{df['radar_type'].unique()[0]}"
        f"_nobs_{x}_distributions_{datesuffix}.{args.ext}"
    )
    plt.close()


def plot_log_nobs_distribution(
    fig,
    df,
    path,
    x="VIS_PT1M_AVG",
    xmin=-10,
    xmax=10,
    ny=20,
    nmin=10,
    vmin=10,
    vmax=1e3,
    x_formatter=m2km_formatter,
    x_unit="km",
    xres=1,
    datesuffix="",
    outpath=Path("."),
    title=True,
):
    """Plot distribution of fraction of available measurements against surface measurement in log-scale.

    Parameters
    ----------
    fig : matplotlib.figure
        Figure object
    df : pandas.DataFrame
        The dataframe with each row representing one paired lidar & radar scan.
    path : str
        Prefix added to filename.
    x : str, optional
        Surface station measurement name, by default "VIS_PT1M_AVG"
    xmin : int, optional
        Lower limit for surface station measurement values, by default -10
    xmax : int, optional
        Upper limit for surface station measurement values, by default 10
    ny : int, optional
        Number of bins in plt.hexbin for fraction of available measurements, by default 20
    nmin : int, optional
        Minimum count of points in bin to plt.hexbin, by default 10
    vmin : int, optional
        Minimum value for colorbar norm, by default 10
    vmax : _type_, optional
        Maximum value for colorbar norm, by default 1e3
    x_formatter : callable, optional
        Matplotlib FuncFormatter to format surface measurement tick labels, by default m2km_formatter
    x_unit : str, optional
        Unit for surface station measurements, by default "km"
    xres : int, optional
        Resolution of surface station measurements, used to calculate bin size for plt.hexbin,
        by default 1
    datesuffix : str, optional
        Date suffix added to the end of filename, by default ""
    outpath : pathlib.Path, optional
        Output directory, by default Path(".")
    title : bool, optional
        Whether to write a title to plot, by default True

    """
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, height_ratios=[1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)

    dfs = df.melt(
        id_vars=["utctime", x],
        value_vars=["frac_valid_bins_lidar", "frac_valid_bins_xband"],
        var_name="instrument",
        value_name="frac_valid",
    )
    dfs.set_index("utctime", inplace=True)

    dfs["instrument"][dfs["instrument"] == "frac_valid_bins_lidar"] = "Lidar"
    dfs["instrument"][dfs["instrument"] == "frac_valid_bins_xband"] = "X-band"

    # import ipdb; ipdb.set_trace()
    N_lidar = dfs[dfs["instrument"] == "Lidar"]["frac_valid"].notna().sum()
    N_xband = dfs[dfs["instrument"] == "X-band"]["frac_valid"].notna().sum()

    hue_norm = mpl.colors.LogNorm(vmin=vmin * 100, vmax=vmax * 100)
    p = ax1.hexbin(
        dfs[(dfs["instrument"] == "Lidar") & (dfs[x] >= xmin)][x],
        dfs[(dfs["instrument"] == "Lidar") & (dfs[x] >= xmin)]["frac_valid"],
        C=np.ones_like(dfs[dfs["instrument"] == "Lidar"]["frac_valid"], dtype=np.float)
        * 100
        / N_lidar,
        bins="log",
        xscale="log",
        norm=hue_norm,
        gridsize=(int((xmax - xmin) / xres), ny),
        mincnt=nmin,
        reduce_C_function=np.sum,
    )
    p = ax2.hexbin(
        dfs[(dfs["instrument"] == "X-band") & (dfs[x] >= xmin)][x],
        dfs[(dfs["instrument"] == "X-band") & (dfs[x] >= xmin)]["frac_valid"],
        C=np.ones_like(dfs[dfs["instrument"] == "X-band"]["frac_valid"], dtype=np.float)
        * 100
        / N_xband,
        bins="log",
        xscale="log",
        norm=hue_norm,
        gridsize=(int((xmax - xmin) / xres), ny),
        mincnt=nmin,
        reduce_C_function=np.sum,
    )
    cbar = plt.colorbar(
        p,
        label="% of cases",
        ax=[ax1, ax2],
        aspect=50,
        orientation="horizontal",
        shrink=0.6,
        extend="max",
        location="top",
        format=ticker.StrMethodFormatter("{x:.1f}"),
    )
    # cbar.axes.xaxis.set_major_formatter(ticker.ScalarFormatter())

    pick_outside_brackets = re.compile("([^[\]]+)(?:$|(?=\[))")
    xname = pick_outside_brackets.findall(names[x])[0]
    title_y = -0.25
    for ax, inst in zip([ax1, ax2], ["(a) Lidar", "(b) X-band radar"]):
        ax.set_xlabel(f"{xname}[{x_unit}]")
        # ax.set_xlabel(f"{names[x]}")
        ax.set_title(inst, y=title_y)
        ax.set_xlim((xmin, xmax))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(5 * xres))
        ax.xaxis.set_major_formatter(x_formatter)
    ax1.set_ylim((0, 1))
    ax1.set_ylabel("Fraction of valid measurements")

    if title:
        fig.suptitle(
            f"{df.index.min():%Y/%m/%d %H:%M} - "
            f"{df.index.max():%Y/%m/%d %H:%M} UTC\n{df['radar_type'].unique()[0]}"
        )
    plt.savefig(
        outpath / f"{path}_{df['radar_type'].unique()[0]}"
        f"_nobs_{x}_distributions_{datesuffix}.{args.ext}"
    )
    plt.close()

names = {
    "utctime": "Time (UTC)",
    "PRIO_PT10M_AVG": "Precipitation intensity (10min average) [mm h$^{-1}$]",
    "stationname": "Station name",
    "CLHB_PT1M_INSTANT": "Cloud base height [km]",
    "VIS_PT1M_AVG": "Horizontal visibility [km]",
}

station_fmisid = 100968
station_name = "Vantaa Helsinki-Vantaan lentoasema"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "rtype", type=str, help="X-band type", choices=["WND-01", "WND-02", "WND-03"]
    )
    argparser.add_argument(
        "datapath", type=str, help="Path where datafiles are located"
    )
    argparser.add_argument("startdate", type=str, help="the startdate (YYYYmmpp )")
    argparser.add_argument("enddate", type=str, help="the enddate (YYYYmmpp )")
    argparser.add_argument("--outpath", type=str, default=".", help="Output path")
    argparser.add_argument(
        "--tol",
        type=int,
        default=1,
        help="Tolerance for merging weather observations, minutes",
    )
    argparser.add_argument(
        "--ext", type=str, default="pdf", help="Figure file extension"
    )
    argparser.add_argument(
        "--var",
        type=str,
        default="none",
        choices=["prio", "vis", "clhb", "none"],
        help="Variable that is calculated",
    )
    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m")
    enddate = (
        datetime.strptime(args.enddate, "%Y%m") + pd.offsets.MonthEnd(0)
    ).to_pydatetime()
    datesuffix = f"{startdate:%Y%m}_{enddate:%Y%m}"

    params = [
        "utctime",
        "stationname",
        # "PRIO_PT10M_AVG",
        # "CLHB_PT1M_INSTANT",
        # "VIS_PT1M_AVG",
    ]
    if args.var == "prio":
        params.append("PRIO_PT10M_AVG")
    elif args.var == "clhb":
        params.append("CLHB_PT1M_INSTANT")
    elif args.var == "vis":
        params.append("VIS_PT1M_AVG")

    rtype = args.rtype
    datapath = args.datapath

    outpath = Path(args.outpath)

    plt.style.use(cfg.STYLE_FILE)

    df_list = []

    paths = Path(datapath).glob("*stats.csv")
    pattern = re.compile("([0-9]{8})_([0-9]{8})_" + f"{rtype}_stats.csv")

    tol = pd.Timedelta(f"{args.tol} minute")
    for csvfn in paths:
        try:
            dates = pattern.findall(csvfn.name)[0]
        except:
            continue

        if len(dates) != 2:
            continue

        start = datetime.strptime(dates[0], "%Y%m%d")
        end = datetime.strptime(dates[1], "%Y%m%d")

        if not (start >= startdate and end <= enddate):
            continue

        try:
            df = pd.read_csv(
                csvfn,
                parse_dates=[
                    "lidar_time",
                    "xband_time",
                ],
            )
        except (FileNotFoundError):
            continue

        df["radar_type"] = rtype
        inst_cols = df.columns

        df_list.append(df)

    df = pd.concat(df_list)
    df1 = df.set_index("lidar_time", drop=False)
    df1.sort_index(inplace=True)

    # Load weather datda
    starttime = df["lidar_time"].min()
    endtime = df["lidar_time"].max()

    wdf = utils.query_Smartmet_station(station_fmisid, starttime, endtime, params)

    # Drop empty rows (if no observations given for some time)
    drop_cols = [c for c in wdf.columns if str(tol.seconds // 60) in c]
    wdf.dropna(how="all", subset=drop_cols, inplace=True)

    wdf1 = wdf.set_index("time")
    wdf1.sort_index(inplace=True)
    df = pd.merge_asof(
        left=wdf1,
        right=df1,
        right_index=True,
        left_index=True,
        direction="nearest",
        tolerance=tol,
    )

    inst_cols = list(inst_cols)
    df.dropna(subset=inst_cols, how="all", inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(subset=inst_cols, inplace=True)

    print(f"Number of cases: {len(df)}")

    # Load mask and count number of observations
    mask = np.loadtxt(cfg.OBS_MASK_PATH)
    nobs = mask.sum()

    df[f"n_lidar_normalized"] = df["nobs_lidar"] / nobs
    df[f"n_xband_normalized"] = df["nobs_xband"] / nobs
    df[f"n_union_normalized"] = df["nobs_union"] / nobs
    df[f"n_common_normalized"] = df["nobs_common"] / nobs

    vmin = 0.0001
    vmax = 0.1

    if args.var == "vis":
        # Horizontal visibility
        final(figsize=(8, 5), constrained_layout=True)(plot_nobs_distribution)(
            df,
            "full",
            x="VIS_PT1M_AVG",
            ny=20,
            xmin=0,
            xmax=8e4,
            xres=25e2,
            vmin=vmin,
            vmax=vmax,
            x_unit="km",
            datesuffix=datesuffix,
            outpath=outpath,
            title=False,
        )

        # final(figsize=(8, 5), constrained_layout=True)(plot_nobs_distribution)(
        #     df,
        #     "full",
        #     x="CLHB_PT1M_INSTANT",
        #     ny=20,
        #     xmin=0,
        #     xmax=8e3,
        #     xres=250,
        #     vmin=vmin,
        #     vmax=vmax,
        #     datesuffix=datesuffix,
        #     outpath=outpath,
        #     title=False,
        # )
    elif args.var == "clhb":
        # Cloud base height
        final(figsize=(8, 5), constrained_layout=True)(plot_nobs_distribution)(
            df,
            "subset",
            x="CLHB_PT1M_INSTANT",
            ny=50,
            x_formatter=m_formatter,
            x_unit="m",
            nmin=1,
            xmin=0,
            xmax=3e3,
            xres=25,
            vmin=vmin,
            vmax=vmax,
            datesuffix=datesuffix,
            outpath=outpath,
            title=False,
        )

        # final(figsize=(8, 5), constrained_layout=True)(plot_log_nobs_distribution)(
        #     df,
        #     "log_subset",
        #     x="CLHB_PT1M_INSTANT",
        #     ny=50,
        #     x_formatter=m_formatter,
        #     x_unit="m",
        #     nmin=1,
        #     xmin=0,
        #     xmax=3e3,
        #     xres=50,
        #     vmin=vmin,
        #     vmax=0.005,
        #     datesuffix=datesuffix,
        #     outpath=outpath,
        #     title=False,
        # )

    elif args.var == "prio":
        # Needs to have tolerance 10
        final(figsize=(8, 5), constrained_layout=True)(plot_nobs_distribution)(
            df,
            "full",
            x="PRIO_PT10M_AVG",
            ny=40,
            xmin=0,
            xmax=4,
            xres=0.015,
            vmin=vmin,
            vmax=vmax,
            nmin=1,
            x_unit="mm h$^{-1}$",
            x_formatter=mmh_formatter,
            datesuffix=datesuffix,
            outpath=outpath,
            title=False,
        )

        # final(figsize=(8, 5), constrained_layout=True)(plot_log_nobs_distribution)(
        #     df,
        #     "log_full",
        #     x="PRIO_PT10M_AVG",
        #     ny=40,
        #     xmin=0.1,
        #     xmax=4,
        #     xres=0.1,
        #     vmin=vmin,
        #     vmax=vmax,
        #     x_unit="mm h$^{-1}$",
        #     x_formatter=mmh_formatter,
        #     datesuffix=datesuffix,
        #     outpath=outpath,
        #     title=False,
        # )
