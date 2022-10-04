"""Run scatterplot gridding for the interval splitted into multiple intervals.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>
"""
import sh
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

script = Path("compute_gridded_lidar_xband.py").resolve()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "rtype",
        type=str,
        help="X-band type",
        choices=["WND-01", "WND-02", "WND-03", "MWS-PPI1_G"],
    )
    argparser.add_argument("startdate", type=str, help="the start month (YYYYmm)")
    argparser.add_argument("enddate", type=str, help="the end month (YYYYmm)")
    argparser.add_argument("outpath", type=str, default=".", help="Output path")
    argparser.add_argument(
        "--month_splits",
        type=int,
        default=3,
        help="Number of intervals a month is splitted to",
    )
    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m")
    enddate = datetime.strptime(args.enddate, "%Y%m")

    months = pd.date_range(startdate, enddate + pd.offsets.MonthEnd(), freq="M")

    processes = []

    for month in months:
        dateintervals = pd.date_range(
            month - pd.offsets.MonthBegin(1),
            month,
            periods=args.month_splits + 1,
        )

        startdays = dateintervals[:-1]
        enddays = dateintervals[1:]

        for start, end in zip(startdays, enddays):

            if end in startdays:
                end -= pd.Timedelta("1 day")

            p = sh.python(
                script,
                args.rtype,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                outpath=args.outpath,
                _bg=True,
            )
            processes.append(p)

    for proc in processes:
        p.wait()
