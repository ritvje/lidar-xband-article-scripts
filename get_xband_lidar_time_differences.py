"""Get time differences between lidar and xband files.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
from datetime import datetime
import logging
import pandas as pd


import utils
import file_utils


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("xband_path", type=str, help="Path to xband directory")
    argparser.add_argument("lidar_path", type=str, help="Path to lidar directory")
    argparser.add_argument("outfile", type=str, help="Output csv file")
    argparser.add_argument(
        "--tasks",
        nargs="+",
        default=["WND-02", "WND-03", "MWS-PPI1_G"],
        help="X-band tasks that are processed",
    )
    argparser.add_argument(
        "--scan_type",
        type=str,
        default="ppi",
        help="Lidar scan type that is matched to",
    )
    argparser.add_argument(
        "--scan_angle",
        type=float,
        default=2.0,
        help="Lidar scan fixed angle that is matched to",
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)

    xband_files = utils.get_sigmet_file_list_by_task(args.xband_path)
    lidar_files = utils.get_lidar_file_list_by_type(args.lidar_path)

    lidar_dict = {}
    for fn in lidar_files[(args.scan_type, args.scan_angle)]:
        tt = file_utils.parse_time_from_filename(
            fn,
            r"WLS400s-113_([0-9_-]{19})_([\w]+)_([\d]+)_([\d]+)m.nc",
            "%Y-%m-%d_%H-%M-%S",
            0,
        )
        lidar_dict[tt] = fn

    data = []
    # Get closest lidar file for each xband file
    for task in args.tasks:
        if task not in xband_files.keys():
            logging.info(f"No files for task {task}, skipping!")
            continue
        logging.info(f"Processing task {task} with {len(xband_files[task])} files")
        for xband_fn in xband_files[task]:
            xband_time = datetime.strptime(xband_fn.split(".")[0], "WRS%y%m%d%H%M%S")

            lidar_time, lidar_fn, timediff = file_utils.find_closest_file(
                xband_time, lidar_dict
            )

            data.append(
                [
                    task,
                    xband_fn,
                    xband_time,
                    lidar_fn,
                    lidar_time,
                    timediff,
                ]
            )

    df = pd.DataFrame(
        data,
        columns=[
            "task",
            "xband_file",
            "xband_time",
            "lidar_file",
            "lidar_time",
            "time_difference_sec",
        ],
    )

    df.to_csv(args.outfile, index=False)
