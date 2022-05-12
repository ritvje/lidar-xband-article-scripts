# Lidar / X-band Doppler velocity article code

This repository contains scripts for creating the figures for the article.


## Python scripts

| Script                                                                       | Description                                                                                                           |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`browse_netcdf_data.py`](browse_netcdf_data.py)                             | Browse lidar netcdf files and write information of the files to a csv file.                                           |
| [`browse_raw_radar_data.py`](browse_raw_radar_data.py)                       | Browse RAW (IRIS/Sigmet) radar files and write information of the files to a csv file.                                |
| [`get_xband_lidar_time_differences.py`](get_xband_lidar_time_differences.py) | Calculate time differences between lidar and X-band radar scans.                                                      |
| [`plot_lidar_xband_case.py`](plot_lidar_xband_case.py)                       | Plot 4-panel case figures of X-band radar and lidar data.                                                             |
| [`plot_lidar_ppi.py`](plot_lidar_ppi.py)                                     | Plot 2-panel figures of lidar PPIs with CNR and Doppler velocity.                                                     |
| [`plot_lidar_figure_list.py`](plot_lidar_figure_list.py)                     | Run [`plot_lidar_ppi.py`](plot_lidar_ppi.py) for a list produced by [`browse_netcdf_data.py`](browse_netcdf_data.py). |

## How to plot figures from the article

### Figures 3 - 5

Run [`make_case_figures.sh`](make_case_figures.sh). Set settings and file root path in the beginning of the script.
