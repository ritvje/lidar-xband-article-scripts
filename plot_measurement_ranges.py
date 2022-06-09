"""Plot a figure of measurement availaibility as function of range.

Looks for files in directory `inpath` called
- measurement_ranges_radar.csv"
- measurement_ranges_lidar.csv

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>
"""

from pathlib import Path
import argparse
import warnings
from datetime import datetime
import pandas as pd
import matplotlib as mlt

mlt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config as cfg

warnings.simplefilter(action="ignore")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    plt.style.use(cfg.STYLE_FILE)

    df_radar = pd.read_csv(inpath / f"measurement_ranges_xband.csv", index_col=0)
    df_lidar = pd.read_csv(inpath / f"measurement_ranges_lidar.csv", index_col=0)

    # Plot percentages
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.set_xlim([0, 15])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    df_radar.index *= 1e-3
    df_lidar.index *= 1e-3

    l3 = ax.plot(df_lidar.index, df_lidar.pct, "k", ls="--", label=f"Lidar", lw=2)
    l4 = ax.plot(df_radar.index, df_radar.pct, "k", ls="-", label=f"X-band radar", lw=2)

    ax.legend()
    ax.grid(which="both", alpha=0.5)
    ax.set_xlim((0, args.maxdist))

    ax.set_ylabel("Fraction of available measurements")
    ax.set_xlabel("Range [km]")

    # ax.set_title(f"Fraction of available measurements")
    fig.savefig(
        outpath / f"lidar_xband_measurement_ranges.pdf",
        dpi=args.dpi,
        bbox_inches="tight",
    )
