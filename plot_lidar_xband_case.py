"""Plot a 4-panel figure of radar dBZ & velocity and lidar CNR & velocity.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
import pyart
from pathlib import Path
import wradlib as wrl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
import arrow

from radar_plotting import plot_utils

plt.style.use("presentation.mplstyle")
background_color = "white"
emph_color = "black"
plt.rcParams.update(
    {
        "axes.facecolor": background_color,
        "text.usetex": False,
        "text.latex.preamble": r"",
        "figure.titleweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.family": "sans-serif",
        #     "font.sans-serif": ["Helvetica"]
        # #     'font.size': 12,
        # #     'font.family': 'Times New Roman',
        # #     'mathtext.fontset': 'cm',
    }
)

pyart.load_config(os.environ.get("PYART_CONFIG"))


def plot_4panel_ppi(
    radar, lidar_sweep, max_dist=15, dpi=300, outpath=Path("."), ext="pdf"
):

    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 10), sharex="col", sharey="row"
    )
    radardisplay = pyart.graph.RadarDisplay(radar)

    radar_time = datetime.strptime(
        radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
    )

    # Plot radar dbz and v
    titles = [
        f"(a) X-band radar $Z_e$",
        f"(b) X-band radar $v_r$",
    ]
    qtys = ["DBZH", "VRAD"]
    for ax, title, qty in zip(axes[0, :].flat, titles, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        cmap, norm = plot_utils.get_colormap(qty)
        cbar_ticks = None
        if norm is None:
            bounds = np.arange(
                plot_utils.QTY_RANGES[qty][0], plot_utils.QTY_RANGES[qty][1] + 0.1, 2.0
            )
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(cmap, len(bounds))
        elif isinstance(norm, mpl.colors.BoundaryNorm):
            cbar_ticks = norm.boundaries

        radardisplay.plot(
            plot_utils.PYART_FIELDS[qty],
            0,
            title="",
            ax=ax,
            axislabels_flag=False,
            colorbar_flag=False,
            cmap=cmap,
            norm=norm,
            zorder=10,
            rasterized=True,
            edgecolor="none",
        )

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(plot_utils.QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            ticks=cbar_ticks,
        )
        cbar.locator = mpl.ticker.MultipleLocator(10 if qty == "DBZH" else 5)
        cbar.update_ticks()
        cbar.set_label(label=plot_utils.COLORBAR_TITLES[qty], weight="bold")
        radardisplay.plot_range_ring(250, ax=ax, lw=0.5, col=emph_color)
        ax.set_title(title, y=-0.12)
    fig.text(
        0.5,
        0.90,
        f"{radar_time:%Y/%m/%d %H:%M:%S} X-band radar 2.0° PPI",
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
    )

    # Plot lidar CNR and velocity
    lidar_time = arrow.get(lidar_sweep["time"].data.min()).datetime

    qtys = ["cnr", "radial_wind_speed"]
    titles = [
        f"(c) Lidar CNR",
        f"(d) Lidar $v_r$",
    ]
    for ax, title, qty in zip(axes[1, :].flat, titles, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        cmap, norm = plot_utils.get_colormap(qty)
        cbar_ticks = None
        if norm is None:
            bounds = np.arange(
                plot_utils.QTY_RANGES[qty][0], plot_utils.QTY_RANGES[qty][1] + 0.1, 2.0
            )
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(cmap, len(bounds))
        elif isinstance(norm, mpl.colors.BoundaryNorm):
            cbar_ticks = norm.boundaries[1::2]

        data = np.ma.array(
            data=lidar_sweep[qty].data, mask=np.zeros(lidar_sweep[qty].data.shape)
        )
        if qty != "cnr":
            np.ma.masked_where(
                lidar_sweep["radial_wind_speed_status"].data == 0, data, copy=False
            )
        wrl.vis.plot_ppi(
            data,
            r=lidar_sweep["range"].data / 1000,
            elev=lidar_sweep["range_gate_length"].data.item(),
            az=(180 + lidar_sweep["azimuth"].data) % 360,
            ax=ax,
            norm=norm,
            cmap=cmap,
            rasterized=True,
            edgecolor="none",
        )
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(plot_utils.QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            ticks=cbar_ticks,
        )
        cbar.locator = mpl.ticker.MultipleLocator(5)
        cbar.update_ticks()
        cbar.set_label(label=plot_utils.COLORBAR_TITLES[qty], weight="bold")

        ax.set_title(title, y=-0.12)
    fig.text(
        0.5,
        0.485,
        f"{lidar_time:%Y/%m/%d %H:%M:%S} Lidar 2.0° PPI",
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
    )

    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
    # x-axis
    for ax in axes[1][:].flat:
        ax.set_xlabel("Distance from site [km]")
        ax.set_title(ax.get_title(), y=-0.22)
        ax.xaxis.set_major_formatter(fmt)

    # y-axis
    for ax in axes.flat[::2]:
        ax.set_ylabel("Distance from site [km]")
        ax.yaxis.set_major_formatter(fmt)

    for ax in axes.flat:
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_aspect(1)
        ax.grid(zorder=0, linestyle="-", linewidth=0.4)

    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.savefig(
        outpath
        / f"data_example_{radar_time:%Y%m%d%H%M%S}_{lidar_time:%Y%m%d%H%M%S}.{ext}",
        dpi=dpi,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("radarfile", type=str, help="the radar file")
    argparser.add_argument("lidarfile", type=str, help="the lidar file")
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

    radar = pyart.io.read_sigmet(args.radarfile)

    lidar = wrl.io.xarray.CfRadial(
        args.lidarfile, flavour="Cf/Radial2", decode_times=False
    )
    sweep = list(lidar.keys())[0]
    lidar_sweep = lidar[sweep]

    plot_4panel_ppi(
        radar,
        lidar_sweep,
        max_dist=args.maxdist,
        dpi=args.dpi,
        outpath=outpath,
        ext=args.ext,
    )
