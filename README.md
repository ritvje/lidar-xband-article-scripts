# Lidar / X-band Doppler velocity article code

This repository contains scripts for creating the figures for the article.


## Python scripts

| Script                                                                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`browse_netcdf_data.py`](browse_netcdf_data.py)                                             | Browse lidar netcdf files and write information of the files to a csv file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [`browse_raw_radar_data.py`](browse_raw_radar_data.py)                                       | Browse RAW (IRIS/Sigmet) radar files and write information of the files to a csv file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [`get_xband_lidar_time_differences.py`](get_xband_lidar_time_differences.py)                 | Calculate time differences between lidar and X-band radar scans.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [`plot_lidar_xband_case.py`](plot_lidar_xband_case.py)                                       | Plot 4-panel case figures of X-band radar and lidar data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [`plot_lidar_ppi.py`](plot_lidar_ppi.py)                                                     | Plot 2-panel figures of lidar PPIs with CNR and Doppler velocity.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [`plot_lidar_figure_list.py`](plot_lidar_figure_list.py)                                     | Run [`plot_lidar_ppi.py`](plot_lidar_ppi.py) for a list produced by [`browse_netcdf_data.py`](browse_netcdf_data.py).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [`compute_measurement_availability.py`](compute_measurement_availability.py)                 | Computes the fraction of available measurements over the given month for lidar and X-band radar measurements. The files for the measurements are searched according to the information given in [`config.py`](config.py) in `RADAR_INFO["fivxt"]` and `LIDAR_INFO["vaisala"]`. Produces also the masks to filter out blocked rays for the analysis.                                                                                                                                                                                                                                                                                                                                                                                     |
| [`plot_ppi_blockage_map.py`](plot_ppi_blockage_map.py)                                       | Plots the PPI availability figure based on output from [`compute_measurement_availability.py`](compute_measurement_availability.py).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) | Computes measurement availability as function of range for binned by different surface station measurements. Currently, bins scans by cloud base height and horizontal visibility if `--tol 1`, by precipitation intensity if `--tol 10`, and by nothing if `--tol 0`. Note that after running the script once, the calculation can be sped up by reading stored data with `--only-read`, even if `--tol` changes. Note that for `--tol 0`, the results differ from the results produced by [`compute_measurement_availability.py`](compute_measurement_availability.py), because here we only consider measurement times where both lidar and radar measurements exist, while that script calculates over all measurements separately. |
| [`plot_measurement_ranges.py`](plot_measurement_ranges.py)                                   | Plot measurement availability as function of range, calculated by [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 0`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [`plot_measurement_ranges_weather.py`](plot_measurement_ranges_weather.py)                   | Plot measurement availability as function of range binned by surface measurements, calculated by [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 1` or ``--tol 10`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## How to plot figures from the article

### Figure 2 (measurement availability PPI)

Run script [`compute_measurement_availability.py`](compute_measurement_availability.py). This produces files with the bin-wise information of available measurements. The PPI is then plotted with [`plot_ppi_blockage_map.py`](plot_ppi_blockage_map.py).

For example:

```bash
python compute_measurement_availability.py 20210501 20211101 --task-name WND-03 --outpath results

# After calculation is done
python plot_ppi_blockage_map.py 20210501 20211101 results --outpath results
```

### Figure 3 (measurement availability as function of range)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 0`, and plot the results with [`plot_measurement_ranges.py`](plot_measurement_ranges.py).

For example:

```bash
python compute_measurement_availability_weather.py 20210501 20211101 --tol 0 --outpath results

# After calculation is done
python plot_measurement_ranges.py results --outpath results
```

### Figures 6 and 10 (measurement availability as function of range for cloud base height and visibility)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 1`

```bash
# If script was run before
python compute_measurement_availability_weather.py 20210501 20211101 --tol 1 --outpath results --only-read

# If running first time
python compute_measurement_availability_weather.py 20210501 20211101 --tol 1 --outpath results

# Plot as function of cloud base height
python plot_measurement_ranges_weather.py results CLHB_PT1M_INSTANT_3000 --log-scale  --outpath results --formatter m2km

# Plot as function of horizontal visibility
python plot_measurement_ranges_weather.py results VIS_PT1M_AVG_75000   --outpath results --formatter m2km

```

### Figure 12 (measurement availability as function of range for precipitation)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 10`

```bash
# If script was run before
python compute_measurement_availability_weather.py 20210501 20211101 --tol 10 --outpath results --only-read

# If running first time
python compute_measurement_availability_weather.py 20210501 20211101 --tol 10 --outpath results

# Plot as function of precipitation intensity
python plot_measurement_ranges_weather.py results PRIO_PT10M_AVG_4  --outpath results --formatter none
```


### Figures 7, 8, and 13 (case figures)

Run [`make_case_figures.sh`](make_case_figures.sh). Set settings and file root path in the beginning of the script.

