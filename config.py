"""Common configurations."""
import os
import inspect
from attrdict import AttrDict

CWD = os.path.dirname(inspect.getfile(inspect.currentframe()))

# All given paths should be full paths
FMI_RADAR_ARCH = "/arch/radar/raw/"
MWSA_DATA_PATH = "/data/MWSA/"

# Lidar and radar paths for given date
LIDAR_PATH = lambda d: f"{MWSA_DATA_PATH}/{d:%Y/%m/%d}/lidar"
XBAND_PATH = lambda d: f"{MWSA_DATA_PATH}/{d:%Y/%m/%d}/xband"

# Multiprocessing with dask
DASK_NWORKERS = 10
DASK_SCHEDULER = "processes"
# DASK_SCHEDULER = "single-threaded"

PLOT_FIGURES = False
PLOT_OUTPUT_FIGURE = False
FIG_OUTPATH = "./figures"
STYLE_FILE = os.path.join(
    CWD,
    "presentation.mplstyle"
    # CWD, ".papers.mplstyle"
)
OBS_MASK_PATH = os.path.join(
    CWD,
    "analysis_14.5km/mask_lidar_202105_202111_WND-03_cnr_250m_14km.txt",
)
POLAR_OBS_MASK_LIDAR_PATH = os.path.join(
    CWD,
    "observation_masks/lidar_obs_pct_20210501_20211031_pct.txt",
    # "mwsa_ppi1_g/lidar_obs_pct_20210501_20211101_pct.txt",
)
POLAR_OBS_MASK_XBAND_PATH = os.path.join(
    CWD,
    "observation_masks/xband_obs_pct_20210501_20211031_pct.txt",
    # "mwsa_ppi1_g/xband_obs_pct_20210501_20211101_pct.txt",
)
POLAR_OBS_MASK_THR = 0.05

DATA_OUTPATH = "/home/users/ritvanen/koodaus/MWSA-WP2/analysis_14.5km_no_filtering"

# Thresholds
CNR_THR = -30.0
RADAR_MEDIAN_FILTER_FACTOR = 1.5
RADAR_MEDIAN_FILTER_WINDOW = 5

LIDAR_INFO = AttrDict(
    {
        "vaisala": {
            "lonlat": (24.87608, 60.28233),
            # "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_([0-9]{3})_200m.nc",
            "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_(438|351)_200m.nc",
            # "filepattern": r"WLS400s-113_([0-9_-]{19})_ppi_(435|350)_200m.nc",
            "timepattern": "%Y-%m-%d_%H-%M-%S",
            "altitude": 35,
        }
    }
)

RADAR_INFO = AttrDict(
    {
        "fivan": {
            # The following are for 2.0 dual-prf
            "start_secs": [129, 142, 142],
            "filepatterns": [
                "%Y%m%d%H%M_VAN.PPI1_G.raw",
                "%Y%m%d%H%M_VAN.PPI2_H.raw",
                "%Y%m%d%H%M_VAN.PPI3_H.raw",
            ],
            "filepattern": r"([0-9]{12})_VAN.PPI(1_G|2_H|3_H).raw",
            "timepattern": "%Y%m%d%H%M",
            "full_name": "Vantaa",
            "lonlat": (24.86902004107833, 60.270620081573725),
            "filepath": "/arch/radar/raw/%Y/%m/%d/iris/raw/VAN",
            "altitude": 83,
        },
        "fivxt": {
            "lonlat": (24.876090008765434, 60.28238005936139),
            "filepattern": r"WRS([0-9]{12}).RAW([0-9A-Z]{4})",
            "timepattern": "%y%m%d%H%M%S",
            "altitude": 35,
        },
    }
)

GRID = AttrDict(
    {
        "res": 250,  # meters
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
