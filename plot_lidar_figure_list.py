"""Plot figures for a list of lidar files.

List should be produced with browse_netcdf_data.py.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
import re
import subprocess
import pandas as pd
from pathlib import Path
import dask.bag as db

PLOT_SCRIPT = Path("plot_lidar_ppi.py").resolve()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("csvlist", type=str, help="The csv lidar file list")
    argparser.add_argument("filepath", type=str, help="Path to lidar files")
    argparser.add_argument(
        "--regex",
        type=str,
        default="WLS400s-113_([0-9_-]{19})_ppi_([0-9]+)_([0-9]+)m.nc",
        help="Filename format as regex for PPI files",
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
    argparser.add_argument(
        "--n_workers",
        type=int,
        default=3,
        help="Number of processes to use (if 1, no multi-processing)",
    )
    args = argparser.parse_args()
    outpath = Path(args.outpath).resolve()
    filepath = Path(args.filepath).resolve()

    ppi_regex = re.compile(args.regex)

    def plot_file(file):
        return_code = subprocess.call(
            (
                f"python {PLOT_SCRIPT} {file} "
                f"--ext {args.ext} "
                f"--outpath {args.outpath} "
                f"--maxdist {args.maxdist} "
                f"--dpi {args.dpi}"
            ),
            shell=True,
        )

    def regex_filter(file):
        if file:
            mo = ppi_regex.match(file)
            if mo:
                return True
            else:
                return False
        else:
            return False

    # Filter files by ppi regex
    df = pd.read_csv(args.csvlist)
    files = df[df["file"].apply(regex_filter)]["file"].tolist()

    # Add rootpath to filenames
    files = [filepath / f for f in files]

    if args.n_workers == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"

    # Run all volumes
    bag = db.from_sequence(files)
    bag.map(plot_file).compute(num_workers=args.n_workers, scheduler=scheduler)
