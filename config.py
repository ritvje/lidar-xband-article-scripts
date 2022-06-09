"""Common configurations."""
import os
import inspect
from attrdict import AttrDict

CWD = os.path.dirname(inspect.getfile(inspect.currentframe()))

# All given paths should be full paths
MWSA_DATA_PATH = "/cephfs/archive/ras_projects/MWSA/"

# Lidar and radar paths for given date
LIDAR_PATH = lambda d: f"{MWSA_DATA_PATH}/{d:%Y/%m/%d}/lidar"
XBAND_PATH = lambda d: f"{MWSA_DATA_PATH}/{d:%Y/%m/%d}/xband"

# Multiprocessing with dask
DASK_NWORKERS = 4
DASK_SCHEDULER = "processes"
# For debugging, disable multiprocessing
# DASK_SCHEDULER = "single-threaded"

# Style file for plots
STYLE_FILE = os.path.join(CWD, "presentation.mplstyle")

# Cartesian mask (bool array) that is used to remove blocked grid points
OBS_MASK_PATH = os.path.join(
    CWD,
    "article_analysis/lidar_cart_mask_20210501_20211130_250m_14km.txt",
)

# Fraction of available measurements in each bin in polar coordinates, used to remove blocked rays
POLAR_OBS_MASK_LIDAR_PATH = os.path.join(
    CWD,
    "article_analysis/lidar_obs_pct_20210501_20211130_pct.txt",
)
POLAR_OBS_MASK_XBAND_PATH = os.path.join(
    CWD,
    "article_analysis/xband_obs_pct_20210501_20211130_pct.txt",
)
# Threshold for blocking
POLAR_OBS_MASK_THR = 0.05

# Thresholds for filtering data when calculating gridded agreement
CNR_THR = -30.0
RADAR_MEDIAN_FILTER_FACTOR = 1.5
RADAR_MEDIAN_FILTER_WINDOW = 5

# Location information for lidar
LIDAR_INFO = AttrDict(
    {
        "vaisala": {
            "lonlat": (24.87608, 60.28233),
            # regex for any PPI scan
            # "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_([0-9]{3})_200m.nc",
            # PPIS with 1000ms accumulation time
            "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_(438|351)_200m.nc",
            # PPIs with 500ms accumulation time
            # "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_(435|350)_200m.nc",
            "timepattern": "%Y-%m-%d_%H-%M-%S",
            "altitude": 35,
        }
    }
)

# Location info for radar
RADAR_INFO = AttrDict(
    {
        # X-band radar at Vaisala site
        "fivxt": {
            "lonlat": (24.876090008765434, 60.28238005936139),
            # regex for files in IRIS format
            "filepattern": r"WRS([0-9]{12}).RAW([0-9A-Z]{4})",
            "timepattern": "%y%m%d%H%M%S",
            "altitude": 35,
        },
    }
)

# Grid information for interpolating to Cartesian coordinates
GRID = AttrDict(
    {
        "res": 250,  # meters
        # Bounding boc
        "bbox": [
            [24.59538688, 60.42089555],
            [25.15454059, 60.14325541],
        ],
        "rlim": 14.5e3,  # for gridding data, from center of grid
    }
)

COLORS = AttrDict(
    {
        "C0": "k",
        "C1": "tab:blue",
        "C2": "tab:orange",
    }
)
