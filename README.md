# Lidar / X-band Doppler velocity article code

This repository contains scripts for creating the figures for the article "Complementarity of Wind Measurements from Co-located X-band Weather Radar and Doppler Lidar" by Ritvanen et al.

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
| [`compute_gridded_lidar_xband.py`](compute_gridded_lidar_xband.py)                           | Grid lidar and X-band radar observations to a Cartesian grid and calculate statistics from paired scans. Configurations given in [`config.py`](config.py). Third argument is the radar task name.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [`plot_gridded_lidar_xband_scatterplot.py`](plot_gridded_lidar_xband_scatterplot.py)         | Plot results from [`compute_gridded_lidar_xband.py`](compute_gridded_lidar_xband.py) calculations. Produces a scatterplot, text file with linear fit statistics, figure with correlation values, and pair-wise scatterplots for lidar and X-band radar variables. (Small modifications allow also plotting gridded measurements with surface measurements.)                                                                                                                                                                                                                                                                                                                                                                             |
| [`plot_surface_meas_distributions.py`](plot_surface_meas_distributions.py)                   | Plot distributions of fraction of available measurements a surface measurement based on output from [`plot_gridded_lidar_xband_scatterplot.py`](plot_gridded_lidar_xband_scatterplot.py)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## How to plot figures from the article

### Figure 2 (measurement availability PPI)

Run script [`compute_measurement_availability.py`](compute_measurement_availability.py). This produces files with the bin-wise information of available measurements. The PPI is then plotted with [`plot_ppi_blockage_map.py`](plot_ppi_blockage_map.py).

For example:

```bash
python compute_measurement_availability.py 20210501 20211130 --task-name WND-03 --outpath results

# After calculation is done
python plot_ppi_blockage_map.py 20210501 20211130 results --outpath results
```

### Figure 3 (scatterplot of gridded measurements)

Run script [`compute_gridded_lidar_xband.py`](compute_gridded_lidar_xband.py) for the interval. Note that it might be better to run this for one month at a time, especially if you want to compare the results per month. Remember to set paths in [`config.py`](config.py). Then plot the results with [`plot_gridded_lidar_xband_scatterplot.py`](plot_gridded_lidar_xband_scatterplot.py).
The script should be run with only one process (in `config.py` `DASK_NWORKERS=1`), as writing the data is not currently safe to do concurrently. The script can be run for longer periods with the helper script `run_scatterplot.py`.

```bash
# Run script, with each month splitted to 4 intervals. Output goes to "results" directory
python run_scatterplot.py WND-03 202105 202111 results --month_splits 4

# When done, plot with this
python plot_gridded_lidar_xband_scatterplot.py WND-03 results 202105 202111 --outpath results
```

### Figure 4 (measurement availability as function of range)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 0`, and plot the results with [`plot_measurement_ranges.py`](plot_measurement_ranges.py).

For example:

```bash
python compute_measurement_availability_weather.py 20210501 20211130 --tol 0 --outpath results --var none

# After calculation is done
python plot_measurement_ranges.py results --outpath results
```

### Figures 5, 9, and 11 (distributions of fraction of available measurements)

Run script [`compute_gridded_lidar_xband.py`](compute_gridded_lidar_xband.py) for the interval, and plot with [`plot_surface_meas_distributions.py`](plot_surface_meas_distributions.py).

```bash
# Run per month (same as for figure 4)
# Run script, with each month splitted to 4 intervals. Output goes to "results" directory
python run_scatterplot.py WND-03 202105 202111 results --month_splits 4

# When done, plot with these
# Horizontal visibility
python plot_surface_meas_distributions.py WND-03 results 202105 202109 --outpath results --tol 1--var vis
python plot_surface_meas_distributions.py WND-03 results 202110 202111 --outpath results --tol 1--var vis

# cloud base height
python plot_surface_meas_distributions.py WND-03 results 202105 202109 --outpath results --tol 1--var clhb
python plot_surface_meas_distributions.py WND-03 results 202110 202111 --outpath results --tol 1--var clhb


# Precipitation intensity
python plot_surface_meas_distributions.py WND-03 results 202105 202111 --outpath results --tol 10 --var prio

```

### Figures 6 and 10 (measurement availability as function of range for cloud base height and visibility)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 1`

```bash
# If script was run before
python compute_measurement_availability_weather.py 202105 202111 --tol 1 --outpath results --only-read --var vis

python compute_measurement_availability_weather.py 202105 202111 --tol 1 --outpath results --only-read --var clhb

# If running first time
python compute_measurement_availability_weather.py 202105 202111 --tol 1 --outpath results --var vis

python compute_measurement_availability_weather.py 202105 202111 --tol 1 --outpath results --var clhb

# Plot as function of cloud base height
python plot_measurement_ranges_weather.py results CLHB_PT1M_INSTANT_3000 --log-scale  --outpath results --formatter m2km

# Plot as function of horizontal visibility
python plot_measurement_ranges_weather.py results VIS_PT1M_AVG_75000   --outpath results --formatter m2km

```

### Figure 12 (measurement availability as function of range for precipitation)

Run script [`compute_measurement_availability_weather.py`](compute_measurement_availability_weather.py) with `--tol 10`

```bash
# If script was run before
python compute_measurement_availability_weather.py 202105 202111 --tol 10 --outpath results --only-read --var prio

# If running first time
python compute_measurement_availability_weather.py 202105 202111 --tol 10 --outpath results --var prio

# Plot as function of precipitation intensity
python plot_measurement_ranges_weather.py results PRIO_PT10M_AVG_4  --outpath results --formatter none
```

### Figures 7, 8, and 13 (case figures)

Run [`make_case_figures.sh`](make_case_figures.sh). Set settings and file root path in the beginning of the script.
