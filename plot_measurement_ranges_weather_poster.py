"""Plot a figure of measurement availaibility as function of range binned by surface measurements.

Looks for files in directory `inpath` called
- measurement_ranges_radar_{var_ext}.csv"
- measurement_ranges_lidar_{var_ext}.csv

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>
"""

from pathlib import Path
import argparse
import warnings
import pandas as pd
import matplotlib as mlt
import numpy as np

mlt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import config as cfg

warnings.simplefilter(action="ignore")


names = {
    "utctime": "Time (UTC)",
    "stationname": "Station name",
    "PRIO_PT10M_AVG": "Precipitation intensity (10min average) [mm h$^{-1}$]",
    "CLHB_PT1M_INSTANT": "Cloud base height [km]",
    "VIS_PT1M_AVG": "Horizontal visibility [km]",
}


@ticker.FuncFormatter
def m2km_formatter(x, pos):
    return f"{x / 1000:.1f}"


def plot_meas_ranges_log_scale(
    lidar_result,
    xband_result,
    lidar_total,
    xband_total,
    wvar,
    cmap="viridis",
    cbar_formatter=None,
    outpath=Path("."),
):
    # Plot results
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "height_ratios": [
                1,
            ]
        },
        figsize=(14, 3),
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
        axes[0, 1].plot(
            xband_result.index.values * 1e-3,
            xband_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )

    l3 = axes[0, 0].plot(
        lidar_total.index.values * 1e-3,
        lidar_total.pct,
        "k",
        ls="dashed",
        label=f"All measurements",
        lw=2,
    )
    l4 = axes[0, 1].plot(
        xband_total.index.values * 1e-3,
        xband_total.pct,
        "k",
        ls="dashed",
        label=f"All measurements",
        lw=2,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        label=names[wvar],
        ax=axes,
        aspect=50,
        orientation="horizontal",
        shrink=0.5,
        extend="max",
        location="top",
        pad=-0.15,
        format=cbar_formatter,
    )
    cbar.set_label(label=names[wvar], weight="normal", labelpad=5)

    cbar.set_ticks(lidar_result.columns)
    # cbar.set_ticklabels(lidar_result.columns)
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("Fraction of valid measurements", fontsize=10)
    props = dict(boxstyle="square", facecolor="white", alpha=0.5, edgecolor="none")
    titles = ["Doppler lidar", "X-band radar"]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel("Range [km]")
        # ax.set_ylim(())

        # Set axis limits
        ax.set_xlim((0, args.maxdist))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_ylim((0, 1.05))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.grid(which="both", alpha=0.5)
        # ax.set_title(title, y=1.0)
        ax.text(
            0.98,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            ha="right",
            bbox=props,
            zorder=100,
        )

    axes[0, 0].legend(
        loc="center",
        bbox_to_anchor=(0.15, 1.15),
        fontsize=10,
        numpoints=2,
    )
    # fig.suptitle(f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}")

    fig.savefig(
        outpath
        / f"measurement_range_{wvar}_{np.max(lidar_result.columns):.0f}.{args.ext}",
        dpi=args.dpi,
        bbox_inches="tight",
    )


def plot_meas_ranges(
    lidar_result,
    xband_result,
    lidar_total,
    xband_total,
    wvar,
    cmap="viridis",
    cbar_formatter=None,
    outpath=Path("."),
):
    # Plot results
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "height_ratios": [
                1,
            ]
        },
        figsize=(14, 3),
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
        axes[0, 1].plot(
            xband_result.index.values * 1e-3,
            xband_result[upperlim],
            c=cmap(norm(upperlim)),
            lw=lw,
        )

    l3 = axes[0, 0].plot(
        lidar_total.index.values * 1e-3,
        lidar_total.pct,
        "k",
        ls="dashed",
        label=f"All measurements",
        lw=2,
    )
    l4 = axes[0, 1].plot(
        xband_total.index.values * 1e-3,
        xband_total.pct,
        "k",
        ls="dashed",
        label=f"All measurements",
        lw=2,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        label=names[wvar],
        ax=axes,
        aspect=50,
        orientation="horizontal",
        shrink=0.5,
        extend="max",
        pad=-0.15,
        location="top",
        format=cbar_formatter,
    )
    cbar.set_label(label=names[wvar], weight="normal")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("Fraction of valid measurements", fontsize=10)
    props = dict(boxstyle="square", facecolor="white", alpha=0.5, edgecolor="none")
    # axes[-1, 0].set_xlabel("Range [km]")
    titles = ["Doppler lidar", "X-band radar"]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel("Range [km]")
        # ax.set_ylim(())

        # Set axis limits
        ax.set_xlim((0, args.maxdist))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_ylim((0, 1.05))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        ax.grid(which="both", alpha=0.5)
        # ax.set_title(title, y=1.0)
        ax.text(
            0.98,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            ha="right",
            bbox=props,
            zorder=100,
        )

    axes[0, 0].legend(
        loc="center",
        bbox_to_anchor=(0.15, 1.15),
        fontsize=10,
        numpoints=2,
    )

    # fig.suptitle(f"{startdate:%Y/%m/%d} - {enddate:%Y/%m/%d}")

    fig.savefig(
        outpath
        / f"measurement_range_{wvar}_{np.max(lidar_result.columns):.0f}.{args.ext}",
        dpi=args.dpi,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "inpath", type=str, help="Path where the input files are located"
    )
    argparser.add_argument(
        "var_ext",
        type=str,
        help="Extension that denotes surface measurement & upper limit of measurements, e.g. CLHB_PT1M_INSTANT_3000. The input files are searched based on this.",
    )
    argparser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Plot colour scale in log-scale.",
    )
    argparser.add_argument(
        "--formatter",
        type=str,
        default="m2km",
        choices=["none", "m2km"],
        help="Formatter for color scale colorbar ticks.",
    )
    argparser.add_argument(
        "--ext",
        type=str,
        default="png",
        choices=["pdf", "png"],
        help="Output plot file format.",
    )
    argparser.add_argument("--outpath", type=str, default=".", help="Output path")
    argparser.add_argument(
        "--maxdist",
        type=float,
        default=15,
        help="Maximum distance plotted in figures in km",
    )
    argparser.add_argument(
        "--dpi", type=int, default=600, help="Dots per inch in figure"
    )
    args = argparser.parse_args()
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    inpath = Path(args.inpath).resolve()
    plt.style.use(cfg.STYLE_FILE)

    df_radar = pd.read_csv(
        inpath / f"measurement_ranges_xband_{args.var_ext}.csv", index_col=0
    ).apply(pd.to_numeric, errors="coerce")
    df_lidar = pd.read_csv(
        inpath / f"measurement_ranges_lidar_{args.var_ext}.csv", index_col=0
    ).apply(pd.to_numeric, errors="coerce")

    # Read data availability from entire campaign
    df_radar_total = pd.read_csv(
        inpath / f"measurement_ranges_xband.csv", index_col=0
    ).apply(pd.to_numeric, errors="coerce")
    df_lidar_total = pd.read_csv(
        inpath / f"measurement_ranges_lidar.csv", index_col=0
    ).apply(pd.to_numeric, errors="coerce")

    df_radar.columns = df_radar.columns.astype(float)
    df_lidar.columns = df_lidar.columns.astype(float)

    if args.formatter == "none":
        formatter = None
    elif args.formatter == "m2km":
        formatter = m2km_formatter
    else:
        raise ValueError(f"Formatter {args.formatter} not available!")

    if args.log_scale:
        plot_meas_ranges_log_scale(
            df_lidar,
            df_radar,
            df_lidar_total,
            df_radar_total,
            "_".join(args.var_ext.split("_")[:-1]),
            cbar_formatter=formatter,
            outpath=outpath,
        )
    else:
        plot_meas_ranges(
            df_lidar,
            df_radar,
            df_lidar_total,
            df_radar_total,
            "_".join(args.var_ext.split("_")[:-1]),
            cbar_formatter=formatter,
            outpath=outpath,
        )
