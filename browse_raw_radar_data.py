"""Browse RAW radar files.

Reads the data, and writes some numbers for each file in a csv file.
The format of the output csv is:

time,
f,
task_name
prt_mode,
prt,
prt_ratio,
pulse_width,
beam_width_h,
n_samples,
duration,
range_resolution,
max_range,
number of sweeps in file,

followed by

sweep_mode,
Nyqvist velocity,
fixed angle,

for each sweep.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import os
import argparse
import pyart
import re
import csv
from datetime import datetime

# Some example regex patterns, store here just in case needed at some point
# regex_pattern = r"KUM100808([1789]{2})([0-9]{4}).RAW([A-Z0-9]{4})"
# regex_pattern = r"KUM100808([0-9]{6}).RAW([A-Z0-9]{4})"
# regex_pattern = r"KER170812([0-9]{6}).RAW([A-Z0-9]{4})"
# regex_pattern = r"([0-9]{12})_IKA.PPI([1-3])_([A-M]).raw"
# regex_pattern = r"WRS([0-9]{12}).RAW([A-Z0-9]{4})"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("path", type=str, help="Path to directory that is processed")
    argparser.add_argument("outfile", type=str, help="Output csv file")
    argparser.add_argument(
        "--regex",
        type=str,
        default="WRS([0-9]{12}).RAW([A-Z0-9]{4})",
        help="Filename format as regex, for example "
        ' "([0-9]{12})_IKA.PPI([1-3])_([A-M]).raw", "WRS([0-9]{12}).RAW([A-Z0-9]{4})"',
    )
    args = argparser.parse_args()

    with open(args.outfile, mode="w") as file:
        writer = csv.writer(
            file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        writer.writerow(
            [
                "time",
                "file",
                "task_name",
                "prt_mode",
                "prt",
                "prt_ratio",
                "pulse_width",
                "beam_width_h",
                "n_samples",
                "duration",
                "lr",
                "nbins",
                "max_range",
                "no_sweeps",
                "sweep_mode",
                "Nyqvist_velocity",
                "fixed_angle",
            ]
        )

    prog = re.compile(args.regex.encode().decode("unicode_escape"))

    fullpath = os.path.abspath(args.path)

    for f in os.listdir(fullpath):
        result = prog.match(f)

        if result is None:
            continue

        try:
            rad = pyart.io.read_sigmet(os.path.join(fullpath, f))
        except (ValueError, OSError):
            print(f"Unable to read {f}!")
            continue

        dtime = datetime.strptime(rad.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
        duration = rad.time["data"].max()
        prt = rad.instrument_parameters["prt"]["data"][0]
        max_range = rad.instrument_parameters["unambiguous_range"]["data"][0]
        beam_width_h = rad.instrument_parameters["radar_beam_width_h"]["data"][0]
        prt_mode = rad.instrument_parameters["prt_mode"]["data"][0].astype(str)
        prt_ratio = rad.instrument_parameters["prt_ratio"]["data"][0]
        pulse_width = rad.instrument_parameters["pulse_width"]["data"][0]
        lr = rad.range["meters_between_gates"][0]
        nbin = rad.range["data"].size
        task = rad.metadata["sigmet_task_name"].decode("utf-8")

        # Get number of pulse samples
        sf = pyart.io.sigmet.SigmetFile(os.path.join(fullpath, f))
        n_samples = sf.product_hdr["product_end"]["samples_used"]
        sf.close()

        row = []
        for sweep in range(rad.nsweeps):
            row.extend(
                [
                    rad.sweep_mode["data"][sweep].astype(str),
                    rad.get_nyquist_vel(sweep),
                    rad.fixed_angle["data"][sweep],
                ]
            )

        with open(args.outfile, mode="a") as file:
            writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            writer.writerow(
                [
                    dtime,
                    f,
                    task,
                    prt_mode,
                    prt,
                    prt_ratio,
                    pulse_width,
                    beam_width_h,
                    n_samples,
                    duration,
                    lr,
                    nbin,
                    max_range,
                    rad.nsweeps,
                ]
                + row
            )
