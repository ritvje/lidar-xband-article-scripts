# Lidar / X-band Doppler velocity article code

This repository contains scripts for creating the figures for the article.


## Python scripts

| Script                                                                       | Description                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`browse_netcdf_data.py`](browse_netcdf_data.py)                             | Browse lidar netcdf files and write information of the files to a csv file.                                                                                                                                                                                                                                                                         |
| [`browse_raw_radar_data.py`](browse_raw_radar_data.py)                       | Browse RAW (IRIS/Sigmet) radar files and write information of the files to a csv file.                                                                                                                                                                                                                                                              |
| [`get_xband_lidar_time_differences.py`](get_xband_lidar_time_differences.py) | Calculate time differences between lidar and X-band radar scans.                                                                                                                                                                                                                                                                                    |
| [`plot_lidar_xband_case.py`](plot_lidar_xband_case.py)                       | Plot 4-panel case figures of X-band radar and lidar data.                                                                                                                                                                                                                                                                                           |
| [`plot_lidar_ppi.py`](plot_lidar_ppi.py)                                     | Plot 2-panel figures of lidar PPIs with CNR and Doppler velocity.                                                                                                                                                                                                                                                                                   |
| [`plot_lidar_figure_list.py`](plot_lidar_figure_list.py)                     | Run [`plot_lidar_ppi.py`](plot_lidar_ppi.py) for a list produced by [`browse_netcdf_data.py`](browse_netcdf_data.py).                                                                                                                                                                                                                               |
| [`compute_measurement_range.py`](compute_measurement_range.py)               | Computes the fraction of available measurements over the given month for lidar and X-band radar measurements. The files for the measurements are searched according to the information given in [`config.py`](config.py) in `RADAR_INFO["fivxt"]` and `LIDAR_INFO["vaisala"]`. Produces also the masks to filter out blocked rays for the analysis. |
| [`plot_ppi_blockage_map.py`](plot_ppi_blockage_map.py)                       | Plots the PPI availability figure based on output from [`compute_measurement_range.py`](compute_measurement_range.py).                                                                                                                                                                                                                              |

## How to plot figures from the article

### Figure 2

Run script [`compute_measurement_range.py`](compute_measurement_range.py). This produces files with the bin-wise information of available measurements. The PPI is then plotted with [`plot_ppi_blockage_map.py`](plot_ppi_blockage_map.py).

For example:

```bash
python compute_measurement_range.py 20210501 20211101 --task-name WND-03 --outpath results

# After calculation is done
python plot_ppi_blockage_map.py 20210501 20211101 results --outpath results
```

### Figures 3 - 5

Run [`make_case_figures.sh`](make_case_figures.sh). Set settings and file root path in the beginning of the script.

