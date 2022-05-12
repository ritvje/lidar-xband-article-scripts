"""Plot a PPI of lidar CNR and velocity.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
import wradlib as wrl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import arrow
from pathlib import Path

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


def plot_lidar_ppi(lidar_sweep, max_dist=15, dpi=300, outpath=Path("."), ext="pdf"):
    """Plot a lidar PPI figure with CNR and Doppler velocity.

    Parameters
    ----------
    lidar_sweep : xarray.Dataset
        The lidar sweep as xarray dataset (read with wradlib.io.xarray.CfRadial).
    max_dist : int, optional
        Maximum distance plotted from instrument, by default 15
    dpi : int, optional
        Dots per inch in output figure, by default 300
    outpath : pathlib.Path, optional
        Output path, by default Path(".")
    ext : str, optional
        File extension, by default "pdf"

    """
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    # Plot lidar CNR and velocity
    lidar_time = arrow.get(lidar_sweep["time"].data.min()).datetime

    qtys = [
        "cnr",
        "radial_wind_speed",
    ]

    titles = [
        f"(a) Lidar CNR",
        f"(b) Lidar $v_r$",
    ]
    for ax, title, qty in zip(axes.flat, titles, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        cmap, norm = plot_utils.get_colormap(qty)
        cbar_ticks = None
        if norm is None:
            # define the bins and normalize
            bounds = np.arange(
                plot_utils.QTY_RANGES[qty][0], plot_utils.QTY_RANGES[qty][1] + 0.1, 0.5
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
        0.985,
        f"{lidar_time:%Y/%m/%d %H:%M:%S} Lidar 2.0° PPI",
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
    )

    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
    # x-axis
    for ax in axes[-1][:].flat:
        ax.set_xlabel("Distance from site [km]")
        ax.set_title(ax.get_title(), y=-0.22)
        ax.xaxis.set_major_formatter(fmt)

    # y-axis
    for ax in axes[:, 0].flat:
        ax.set_ylabel("Distance from site [km]")
        ax.yaxis.set_major_formatter(fmt)

    for ax in axes.flat:
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_aspect(1)
        ax.grid(zorder=0, linestyle="-", linewidth=0.4)

    fig.subplots_adjust(wspace=0, hspace=0.2)

    outpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outpath / f"lidar_{lidar_time:%Y%m%d%H%M%S}.{ext}",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        help="Maximum distance in plotted figures in km",
    )
    argparser.add_argument(
        "--dpi", type=int, default=300, help="Dots per inch in figure"
    )
    args = argparser.parse_args()
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    lidar = wrl.io.xarray.CfRadial(
        args.lidarfile, flavour="Cf/Radial2", decode_times=False
    )
    sweep = list(lidar.keys())[0]
    lidar_sweep = lidar[sweep]

    plot_lidar_ppi(
        lidar_sweep, max_dist=args.maxdist, dpi=args.dpi, outpath=outpath, ext=args.ext
    )
