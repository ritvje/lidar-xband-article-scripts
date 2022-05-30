"""Plot a 2-panel figure of radar and lidar availability PPIs.

Looks for files in directory `inpath` called
- xband_obs_pct_{startdate}_{enddate}_pct.txt
- xband_obs_pct_{startdate}_{enddate}_range.txt
- xband_obs_pct_{startdate}_{enddate}_azimuth.txt
- lidar_obs_pct_{startdate}_{enddate}_pct.txt
- lidar_obs_pct_{startdate}_{enddate}_range.txt
- lidar_obs_pct_{startdate}_{enddate}_azimuth.txt

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mlt

mlt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs

plt.style.use("../presentation.mplstyle")


from radar_plotting import plotting


import contextily as ctx
from pathlib import Path

centerpoint = (24.87608, 60.28233)

airport_aws = (24.95675, 60.32670)


@mlt.ticker.FuncFormatter
def m2km_formatter(x, pos):
    return f"{x / 1000:.0f}"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("startdate", type=str, help="the startdate (YYYYmm)")
    argparser.add_argument("enddate", type=str, help="the enddate (YYYYmm)")
    argparser.add_argument(
        "inpath", type=str, help="Path where the input files are located"
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
        "--dpi", type=int, default=300, help="Dots per inch in figure"
    )
    args = argparser.parse_args()
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    inpath = Path(args.inpath).resolve()
    startdate = datetime.strptime(args.startdate, "%Y%m%d")
    enddate = datetime.strptime(args.enddate, "%Y%m%d")

    pct_xband = np.loadtxt(
        inpath / f"xband_obs_pct_{startdate:%H%m%d}_{enddate:%Y%m%d}_pct.txt"
    )
    xband_rr = np.loadtxt(
        inpath / f"xband_obs_pct_{startdate:%H%m%d}_{enddate:%Y%m%d}_range.txt"
    )
    xband_az = np.loadtxt(
        inpath / f"xband_obs_pct_{startdate:%H%m%d}_{enddate:%Y%m%d}_azimuth.txt"
    )

    pct_lidar = np.loadtxt(
        inpath / f"lidar_obs_pct_{startdate:%Y%m%d}_{enddate:%Y%m%d}_pct.txt"
    )
    lidar_rr = np.loadtxt(
        inpath / f"lidar_obs_pct_{startdate:%Y%m%d}_{enddate:%Y%m%d}_range.txt"
    )
    lidar_az = np.loadtxt(
        inpath / f"lidar_obs_pct_{startdate:%Y%m%d}_{enddate:%Y%m%d}_azimuth.txt"
    )

    outfn = os.path.join(outpath, f"meas_pct_map.{args.ext}")

    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    fig = plt.figure(figsize=(12, 10))

    ax_lidar, fig, aeqd, ext = plotting.axes_with_background_map(
        centerpoint, 15, 10, fig=fig, no_map=True, map="toner-line", ncols=2, index=1
    )
    ctx.add_basemap(
        ax_lidar, crs=aeqd, zorder=9, zoom=11, source=ctx.providers.Stamen.TonerLite
    )

    p = plotting.plot_ppi(
        ax_lidar,
        pct_lidar,
        lidar_az,
        lidar_rr,
        rasterized=True,
        vmin=0,
        vmax=1,
        cmap="viridis",
        zorder=100,
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        edgecolor="none",
    )
    ax_lidar.scatter(
        *airport_aws,
        s=75,
        transform=ccrs.PlateCarree(),
        zorder=110,
        label="Helsinki Airport",
        marker="X",
        color="k",
    )

    ax_radar, fig, aeqd, ext = plotting.axes_with_background_map(
        centerpoint,
        15,
        10,
        fig=fig,
        no_map=True,
        map="toner-line",
        sharey=ax_lidar,
        ncols=2,
        index=2,
    )
    ctx.add_basemap(
        ax_radar, crs=aeqd, zorder=9, zoom=11, source=ctx.providers.Stamen.TonerLite
    )

    p = plotting.plot_ppi(
        ax_radar,
        pct_xband,
        xband_az,
        xband_rr,
        rasterized=True,
        vmin=0,
        vmax=1,
        cmap="viridis",
        zorder=100,
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        edgecolor="none",
    )
    ax_radar.scatter(
        *airport_aws,
        s=75,
        transform=ccrs.PlateCarree(),
        zorder=110,
        label="Helsinki Airport",
        marker="X",
        color="k",
    )

    cax = inset_axes(ax_radar, bbox_transform=ax_radar.transAxes, **cbar_ax_kws)
    cbar = plt.colorbar(p, orientation="vertical", cax=cax, ax=None)
    cbar.set_label("Fraction", weight="bold")
    cbar.ax.tick_params(labelsize=12)

    for ax, title in zip([ax_lidar, ax_radar], ["(a) Lidar", "(b) X-band radar"]):

        plotting.set_ticks_km(
            ax,
            [
                -args.maxdist * 1e3,
                args.maxdist * 1e3,
                -args.maxdist * 1e3,
                args.maxdist * 1e3,
            ],
            16,
            16,
        )
        # x-axis
        ax.set_xlabel("Distance from site [km]", weight="bold", size="medium")
        ax.set_title(title, y=-0.15, size="large")
        ax.xaxis.set_major_formatter(m2km_formatter)

        # y-axis
        ax.set_ylabel("Distance from site [km]", weight="bold", size="medium")
        ax.yaxis.set_major_formatter(m2km_formatter)

        ax.set_xlim([-args.maxdist * 1e3, args.maxdist * 1e3])
        ax.set_ylim([-args.maxdist * 1e3, args.maxdist * 1e3])

        ax.tick_params(axis="both", which="major", labelsize="small")

        ax.set_aspect(1)

    fig.savefig(outfn, dpi=args.dpi, bbox_inches="tight")
