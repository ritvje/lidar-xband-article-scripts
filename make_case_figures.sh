#!/usr/bin/env bash

PLOT_SCRIPT="plot_lidar_xband_case.py"
EXT=png
MAXDIST=15
DPI=300
OUTPATH="./figures"

ROOTPATH=""

# Case 1: May 17th 2021 at 11:13 UTC
python "$PLOT_SCRIPT" \
    "${ROOTPATH}/2021/05/17/xband/WRS210517111308.RAWTMWK" \
    "${ROOTPATH}/2021/05/17/lidar/WLS400s-113_2021-05-17_11-13-06_ppi_351_200m.nc" \
    --ext $EXT --maxdist $MAXDIST --dpi $DPI --outpath $OUTPATH

# Case 2: August 28th 2021 at 12:28 UTC
python "$PLOT_SCRIPT" \
    "${ROOTPATH}/2021/08/28/xband/WRS210828122808.RAW5H7L" \
    "${ROOTPATH}/2021/08/28/lidar/WLS400s-113_2021-08-28_12-28-05_ppi_438_200m.nc" \
    --ext $EXT --maxdist $MAXDIST --dpi $DPI --outpath $OUTPATH

# CASE 3: June 16th 2021 at 16:28 UTC
python "$PLOT_SCRIPT" \
    "${ROOTPATH}/2021/06/17/xband/WRS210617162809.RAWULA9" \
    "${ROOTPATH}/2021/06/17/lidar/WLS400s-113_2021-06-17_16-28-06_ppi_351_200m.nc" \
    --ext $EXT --maxdist $MAXDIST --dpi $DPI --outpath $OUTPATH


