"""Utility functions used in data processing.

Includes functions
- query_Smartmet_station
- create_grid
- lidar_to_cart
- radar_to_cart
- lidar_spherical_to_xyz

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import numpy as np
import pyproj
import wradlib as wrl
import requests
import pandas as pd


def query_Smartmet_station(
    fmisid, starttime, endtime, params, payload_params=None, url=None
):
    """Query SmartMet timeseries data from a weather station.

    Gets a timeseries from the station defined by `fmisid`.

    Parameters
    ----------
    fmisid : int
        FMISID for the station.
    starttime : datetime.datetime
        Start time for query.
    endtime : datetime.datetime
        End time for query.
    params : list
        List of queried parameters.
    payload_params : dict
        Extra parameters passed to query.
    url : str
        The base url to the query. Default is the internal FMI url.

    Returns
    -------
    df : pandas.DataFrame
        The results in a DataFrame, with UTC time of observations as
        index and parameter names as columns.

    """
    # Get weather information from SmartMet
    if url is None:
        url = "http://smartmet.fmi.fi/timeseries"
    if "timeseries" not in url:
        url += "/timeseries"

    if "utctime" not in params:
        params.append("utctime")

    if payload_params is None:
        payload_params = {}

    payload = {
        "producer": "observations_fmi",
        "fmisid": fmisid,  # Kumpula station
        "param": ",".join(params),
        "starttime": starttime,
        "endtime": endtime,
        "format": "json",
        "tz": "UTC",
        "precision": "double",
        **payload_params,
    }
    r = requests.get(url, params=payload)

    df = pd.DataFrame(r.json())

    df["time"] = pd.to_datetime(df["utctime"])

    # Transform all possible columns to numeric
    columns = [c for c in df.columns if "time" not in c]
    for column in columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    return df


def create_grid(corners, xres, yres, t_crs="EPSG:3067"):
    """Create a regular grid in the target CRS.

    Parameters
    ----------
    corners : list
        The corners of the grid in lat/lon in the order:
        top left, bottom right.
    xres : float
        The x resolution of the grid, in target CRS units.
    yres : float
        The y resolution of the grid, in target CRS units.
    t_crs : str
        The target coordinate reference system,
        a string that can be used to initialise a coordinate reference
        object.

    Returns
    -------
    X : np.ndarray
        The x coordinates of the grid.
    Y : np.ndarray
        The y coordinates of the grid.

    """
    # Set up projections
    proj_latlon = pyproj.Proj("epsg:4326")
    proj_target = pyproj.Proj(t_crs)  # metric

    # Create corners of rectangle to be transformed to a grid
    # Top left
    topleft = corners[0]
    bottomright = corners[1]
    topleft_m = pyproj.transform(
        proj_latlon, proj_target, topleft[0], topleft[1], always_xy=True
    )
    bottomright_m = pyproj.transform(
        proj_latlon, proj_target, bottomright[0], bottomright[1], always_xy=True
    )

    xx = np.arange(topleft_m[0], bottomright_m[0], xres)
    yy = np.arange(bottomright_m[1], topleft_m[1], yres)

    X, Y = np.meshgrid(xx, yy)
    return X, Y, proj_target


def lidar_to_cart(
    polar_data,
    theta,
    r,
    elev,
    lidar_lonlat,
    xgrid,
    ygrid,
    zlims=[0, 1e3],
    grid_proj4=None,
    ipol_method="nearest",
    r_e=6371288,
    rlim=None,
):
    """Reproject radar polar data to cartesian grid.

    Parameters
    ----------
    polar_data : numpy.ma.ndarray
        The data that is reprojected. The first axis is the azimuth
        angles and the second is the range. Needs to be a masked array.
    theta : float
        The azimuth values of the data.
    r : float
        The ranges along line-of-sight of the data.
    radar_lonlat : np.ndarray
        The lon, lat coordinates of the radar.
    xgrid : numpy.ndarray
        The new x coordinates for all grid points (the output from np.meshgrid).
    ygrid : numpy.ndarray
        The new y coordinates for all grid points (the output from np.meshgrid).
    zlims : array_like
        The altitude limits for the radar beams that are interpolated to grid.
    grid_proj4 : str
        Proj4 string of the grid. If None, the aeqd projection of the radar is used.
    ipol_method : string
        The interpolation method, should correspond to a wradlib
        interpolation method.
    r_e : float
        The radius of Earth. Default 6371288 meters.
    rlim : float
        The maximum radius fro interpolated data.

    Returns
    -------
    gridded : numpy.ma.ndarray
        The data reprojected onto new grid.
    rad : osgeo.osr.SpatialReference
        The projection of the grid.
    """
    # Coordinates of lidar bins in cartesian
    coords, lid = lidar_spherical_to_xyz(
        elev, r, theta, lidar_lonlat, r_e=r_e, squeeze=True
    )

    # Project to new grid
    if grid_proj4 is not None:
        proj = wrl.georef.projection.proj4_to_osr(grid_proj4)
        coords = wrl.georef.reproject(
            coords, projection_source=lid, projection_target=proj
        )
        rproj = proj
    else:
        rproj = lid

    xlidar = coords[..., 0]
    ylidar = coords[..., 1]
    zlidar = coords[..., 2]

    # Mask points where beam is too high or low
    zcond = (zlidar < zlims[0]) | (zlidar > zlims[1])
    polar_data[zcond] = np.nan

    xy = np.concatenate([xlidar.ravel()[:, None], ylidar.ravel()[:, None]], axis=1)

    # Define grid
    # grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((xgrid.ravel(), ygrid.ravel())).transpose()

    # select interpolate method
    if ipol_method.lower() == "nearest":
        method = wrl.ipol.Nearest
    else:
        raise ValueError(f"Interpolation method'{ipol_method} not implemented!")

    if rlim is None:
        rlim = np.max(r)

    polar_data.set_fill_value(np.nan)
    # Interpolate to new grid
    gridded = wrl.comp.togrid(
        xy,
        grid_xy,
        rlim,
        np.array([xlidar.mean(), ylidar.mean()]),
        polar_data.ravel().filled(),
        method,
    )
    gridded = np.ma.masked_invalid(gridded).reshape((xgrid.shape[0], xgrid.shape[1]))
    gridded.set_fill_value(np.nan)
    return gridded, rproj


def radar_to_cart(
    polar_data,
    theta,
    r,
    elev,
    radar_lonlat,
    xgrid,
    ygrid,
    zlims=[0, 1e3],
    grid_proj4=None,
    ipol_method="nearest",
    r_e=6371288,
    rlim=None,
):
    """Reproject radar polar data to cartesian grid.

    Parameters
    ----------
    polar_data : numpy.ma.ndarray
        The data that is reprojected. The first axis is the azimuth
        angles and the second is the range. Needs to be a masked array.
    theta : float
        The azimuth values of the data.
    r : float
        The ranges along line-of-sight of the data.
    radar_lonlat : np.ndarray
        The lon, lat coordinates of the radar.
    xgrid : numpy.ndarray
        The new x coordinates for all grid points (the output from np.meshgrid).
    ygrid : numpy.ndarray
        The new y coordinates for all grid points (the output from np.meshgrid).
    zlims : array_like
        The altitude limits for the radar beams that are interpolated to grid.
    grid_proj4 : str
        Proj4 string of the grid. If None, the aeqd projection of the radar is used.
    ipol_method : string
        The interpolation method, should correspond to a wradlib
        interpolation method.
    r_e : float
        The radius of Earth. Default 6371288 meters.
    rlim : float
        The maximum radius for interpolated data.

    Returns
    -------
    gridded : numpy.ma.ndarray
        The data reprojected onto new grid.
    rad : osgeo.osr.SpatialReference
        The projection of the grid.
    """
    # Coordinates of radar bins in cartesian
    coords, rad = wrl.georef.spherical_to_xyz(
        r, theta, elev, radar_lonlat, re=r_e, squeeze=True
    )

    # Project to new grid
    if grid_proj4 is not None:
        proj = wrl.georef.projection.proj4_to_osr(grid_proj4)
        coords = wrl.georef.reproject(
            coords, projection_source=rad, projection_target=proj
        )
        rproj = proj
    else:
        rproj = rad

    xradar = coords[..., 0]
    yradar = coords[..., 1]
    zradar = coords[..., 2]

    # Mask points where beam is too high or low
    zcond = (zradar < zlims[0]) | (zradar > zlims[1])
    polar_data[zcond] = np.nan

    xy = np.concatenate([xradar.ravel()[:, None], yradar.ravel()[:, None]], axis=1)

    # Define grid
    # grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((xgrid.ravel(), ygrid.ravel())).transpose()

    # select interpolate method
    if ipol_method.lower() == "nearest":
        method = wrl.ipol.Nearest
    else:
        raise ValueError(f"Interpolation method'{ipol_method} not implemented!")

    if rlim is None:
        rlim = np.max(r)

    polar_data.set_fill_value(np.nan)
    # Interpolate to new grid
    gridded = wrl.comp.togrid(
        xy,
        grid_xy,
        rlim,
        np.array([xradar.mean(), yradar.mean()]),
        polar_data.ravel().filled(),
        method,
    )
    gridded = np.ma.masked_invalid(gridded).reshape((xgrid.shape[0], xgrid.shape[1]))
    gridded.set_fill_value(np.nan)
    return gridded, rproj


def lidar_spherical_to_xyz(
    elev, r, azimuth, sitecoords=(0, 0, 0), r_e=6371288, squeeze=True, strict_dims=True
):
    """Compute the geographical coordinates for the lidar beam.

    Code taken modified slightly from
    wradlib.georef.polar.spherical_to_xyz
    (see https://github.com/wradlib/wradlib/blob/master/wradlib/georef/polar.py#L28)

    Parameters
    ----------
    elev : float
        The elevation angle of the beam.
    r : np.array
        The distances along the beam.
    azimuth : np.array
        The azimuth angles of the beams.
    sitecoords : np.array-like
        The geographical coordinates for the lidar site as (x, y, z).
        If not given, assumes zero.

    Returns
    -------
    xyz : np.array
        The coordinates for the bins, shape ().

    """
    # if site altitude is present, use it, else assume it to be zero
    try:
        centalt = sitecoords[2]
    except IndexError:
        centalt = 0.0

    # if no radius is given, get the approximate radius of the WGS84
    # ellipsoid for the site's latitude
    if r_e is None:
        r_e = wrl.georef.projection.get_earth_radius(sitecoords[1])
        # Set up aeqd-projection sitecoord-centered, wgs84 datum and ellipsoid
        # use world azimuthal equidistant projection
        projstr = (
            "+proj=aeqd +lon_0={lon:f} +x_0=0 +y_0=0 +lat_0={lat:f} "
            + "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            + ""
        ).format(lon=sitecoords[0], lat=sitecoords[1])

    else:
        # Set up aeqd-projection sitecoord-centered, assuming spherical earth
        # use Sphere azimuthal equidistant projection
        projstr = (
            "+proj=aeqd +lon_0={lon:f} +lat_0={lat:f} +a={a:f} "
            "+b={b:f} +units=m +no_defs"
        ).format(lon=sitecoords[0], lat=sitecoords[1], a=r_e, b=r_e)

    rad = wrl.georef.projection.proj4_to_osr(projstr)

    r = np.asanyarray(r)
    theta = np.asanyarray(elev)
    phi = np.asanyarray(azimuth)

    if r.ndim:
        r = r.reshape((1,) * (3 - r.ndim) + r.shape)

    if phi.ndim:
        phi = phi.reshape((1,) + phi.shape + (1,) * (2 - phi.ndim))

    if not theta.ndim:
        theta = np.broadcast_to(theta, phi.shape)

    dims = 3
    if not strict_dims:
        if phi.ndim and theta.ndim and (theta.shape[0] == phi.shape[1]):
            dims -= 1
        if r.ndim and theta.ndim and (theta.shape[0] == r.shape[2]):
            dims -= 1

    if theta.ndim and phi.ndim:
        theta = theta.reshape(theta.shape + (1,) * (dims - theta.ndim))

    z = r * np.sin(np.radians(theta)) + centalt
    dist = r * np.cos(np.radians(theta))

    if (not strict_dims) and phi.ndim and r.ndim and (r.shape[2] == phi.shape[1]):
        z = np.squeeze(z)
        dist = np.squeeze(dist)
        phi = np.squeeze(phi)

    x = dist * np.cos(np.radians(90 - phi))
    y = dist * np.sin(np.radians(90 - phi))

    if z.ndim:
        z = np.broadcast_to(z, x.shape)

    xyz = np.stack((x, y, z), axis=-1)

    if xyz.ndim == 1:
        xyz.shape = (1,) * 3 + xyz.shape
    elif xyz.ndim == 2:
        xyz.shape = (xyz.shape[0],) + (1,) * 2 + (xyz.shape[1],)

    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz, rad
