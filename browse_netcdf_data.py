"""Browse netCDF4 (lidar/CfRadial) files.

Reads the data, and writes some numbers for the first sweep of each file in a csv file.
The format of the output csv is:

time,
f,
scan_type
scan_number,
sweep_name,
fixed angle,
duration,
range_resolution,
max_range,

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
import argparse
import pyart
import re
import csv
import pandas as pd
from datetime import datetime, timedelta
from wradlib.io.xarray import CfRadial
from netCDF4 import Dataset

# Some example regex patterns, store here just in case needed at some point
# regex_pattern = r"KUM100808([1789]{2})([0-9]{4}).RAW([A-Z0-9]{4})"
# regex_pattern = r"KUM100808([0-9]{6}).RAW([A-Z0-9]{4})"
# regex_pattern = r"KER170812([0-9]{6}).RAW([A-Z0-9]{4})"
# regex_pattern = r"([0-9]{12})_IKA.PPI([1-3])_([A-M]).raw"
# regex_pattern = r"WRS([0-9]{12}).RAW([A-Z0-9]{4})"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument(
        "path", type=str, help="Path to directory that is processed")
    argparser.add_argument(
        "outfile", type=str, help="Output csv file")
    argparser.add_argument(
        "--regex", type=str,
        default="WLS400s-113_([0-9_-]{19})_([a-z]+)_([0-9]+)_([0-9]+)m.nc",
        help='Filename format as regex')
    args = argparser.parse_args()

    with open(args.outfile, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        writer.writerow([
            "time",
            "file",
            "scan_type",
            "scan_number",
            "sweep_name",
            "fixed_angle",
            "duration",
            "range_resolution",
            "max_range",
        ])

    prog = re.compile(args.regex.encode().decode('unicode_escape'))

    fullpath = os.path.abspath(args.path)

    for f in os.listdir(fullpath):
        result = prog.match(f)

        if result is None:
            continue

        scan_type = result.groups()[1]
        scan_no = result.groups()[2]

        try:
            cf2 = CfRadial(
                os.path.join(fullpath, f),
                flavour="Cf/Radial2",
                decode_times=False)
        except Exception as e:
            print(f"Unable to read {f}: {e}!")
            continue

        sweep = list(cf2.keys())[0]
        ref_time = datetime.strptime(cf2[sweep].time_reference.data.item(),
                                     "%Y-%m-%dT%H:%M:%SZ")
        dtime = ref_time + timedelta(seconds=cf2[sweep].time.data.min())
        duration = cf2[sweep].time.data.max() - cf2[sweep].time.data.min()
        max_range = cf2[sweep].range.data.max()
        lr = float(cf2[sweep].range.attrs["meters_between_gates"])
        fixed_angle = list(cf2.sweeps)[0][1]

        with open(args.outfile, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)

            writer.writerow([
                dtime,
                f,
                scan_type,
                scan_no,
                sweep,
                fixed_angle,
                duration,
                lr,
                max_range,
            ])
