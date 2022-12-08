import sys
import os
import math
import numpy as np


def in_china(lng, lat):
    return lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55


def _transform_latitude(lng, lat):
    ret = (
        -100.0
        + 2.0 * lng
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lng * lat
        + 0.2 * np.sqrt(np.fabs(lng))
    )
    ret += (
        (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 * np.sin(2.0 * lng * np.pi))
        * 2.0
        / 3.0
    )
    ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
            np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (
        (160.0 * np.sin(lat / 12.0 * np.pi) + 320 * np.sin(lat * np.pi / 30.0))
        * 2.0
        / 3.0
    )
    return ret


def _transform_longitude(lng, lat):
    ret = (
        300.0
        + lng
        + 2.0 * lat
        + 0.1 * lng * lng
        + 0.1 * lng * lat
        + 0.1 * np.sqrt(np.fabs(lng))
    )
    ret += (
        (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 * np.sin(2.0 * lng * np.pi))
        * 2.0
        / 3.0
    )
    ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
            np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (
        (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 * np.sin(lng / 30.0 * np.pi))
        * 2.0
        / 3.0
    )
    return ret


def _transform(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = _transform_latitude(lng - 105.0, lat - 35.0)
    dlng = _transform_longitude(lng - 105.0, lat - 35.0)
    radlat = np.radians(lat)
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    lat += dlat
    lng += dlng

    return lng, lat


def wgs2gcj(lng, lat):
    '''
    input & return `lng`, `lat` in degrees
    note that `lng` and `lat` can be vector
    '''
    assert in_china(lng, lat), 'not in china'
    return _transform(lng, lat)


def gcj2wgs(lng, lat):
    '''
    input & return `lng`, `lat` in degrees
    note that `lng` and `lat` can be vector
    '''
    newlng, newlat = _transform(lng, lat)
    return lng*2 - newlng, lat*2-newlat


def wgs_distance(lng1: float, lat1: float, lng2: float, lat2: float):
    '''
    input degrees, return distance on sphere
    this should not be very accurate.
    '''
    earth_radius_km = 6371.0
    lng1r = np.radians(lng1)
    lat1r = np.radians(lat1)
    lng2r = np.radians(lng2)
    lat2r = np.radians(lat2)
    u = np.sin((lat2r - lat1r) / 2)
    v = np.sin((lng2r - lng1r) / 2)
    return (
        1000.0
        * 2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(u ** 2 + np.cos(lat1r) * np.cos(lat2r) * v * v))
    )


def ll2Rne(lon: float, lat: float):
    '''
    get rotation at (`lon`, `lat`), in degrees
    '''
    lon, lat = math.radians(lon), math.radians(lat)
    c_lat = np.cos(lat)
    s_lat = np.sin(lat)
    c_lon = np.cos(lon)
    s_lon = np.sin(lon)
    return np.array(
        [
            [-s_lon, -s_lat * c_lon, c_lat * c_lon],
            [c_lon, -s_lat * s_lon, c_lat * s_lon],
            [0, c_lat, s_lat],
        ],
        dtype=np.float64,
    )


def lla2xyz(lon, lat, alt):
    '''
    `lon`, `lat` in degrees, alt in meters. return x, y, z
    note that input can be vector
    '''
    lonr, latr = np.radians(lon), np.radians(lat)
    RADIUS = 6378137  # semi-major axis
    e = 1 / 298.257223563  # flattening
    N = RADIUS / np.sqrt(np.cos(latr) * np.cos(latr) + (1 - e)
                         ** 2 * np.sin(latr) * np.sin(latr))
    h = N + alt

    x = h * np.cos(latr) * np.cos(lonr)
    y = h * np.cos(latr) * np.sin(lonr)
    z = (N * (1 - e) ** 2 + alt) * np.sin(latr)
    return x, y, z


def xyz2lla(x, y, z):
    '''
    `x`, `y`, `z` can be vector, 
    return (lon, lat, alt) in degrees, degrees, meters
    '''
    a = 6378137.0  # semi-major axis
    e = 1 / 298.257223563  # flattening
    e1s = 0.00669437999013  # 1st-eccentricity

    L = np.arctan2(y, x)
    tmpB = z / np.sqrt(x ** 2 + y ** 2)
    B = np.arctan(tmpB / (1 - e) ** 2)

    maxItrNum = 500

    oldB = B
    err = 10
    thre = 1e-15
    itrNum = 0

    while err >= thre:
        R1 = x / (np.cos(L) * np.cos(B))
        R2 = a / np.sqrt(np.cos(B) * np.cos(B) + (1 - e1s)
                         * np.sin(B) * np.sin(B))
        B = np.arctan(tmpB * R1 / (R1 - R2 * e1s))
        err = abs(oldB - B)
        oldB = B
        itrNum += 1
        if itrNum > maxItrNum:
            import warnings
            warnings.warn(
                f"max itreration number({maxItrNum}) meets in convertion from x-y-z to lla")
            break
    H = R1 - R2
    lat = np.degrees(B)
    lon = np.degrees(L)
    alt = H
    return lon, lat, alt


def xyz2enu(xyz: np.ndarray, anchor) -> np.ndarray:
    '''
    xyz: (3, ) vector or (N, 3) array
    anchor: [lon, lat, (0)] in degress
    '''
    lon, lat = anchor[0], anchor[1]
    alt = anchor[2] if len(anchor) == 3 else 0
    R = ll2Rne(lon, lat)
    x0, y0, z0 = lla2xyz(lon, lat, alt)

    return (xyz - np.r_[x0, y0, z0]) @ R


def enu2xyz(enu: np.ndarray, anchor) -> np.ndarray:
    '''
    enu: (3, ) vector or (N, 3) array
    anchor: [lon, lat, (0)] in degress
    '''
    lon, lat = anchor[0], anchor[1]
    alt = anchor[2] if len(anchor) == 3 else 0
    R = ll2Rne(lon, lat)
    x0, y0, z0 = lla2xyz(lon, lat, alt)

    return enu @ R.T + np.r_[x0, y0, z0]


def enu2lla(enu: np.ndarray, anchor):
    '''
    xyz2lla \circ enu2xyz
    enu should be (N, 3) 2d array
    '''
    xyz = enu2xyz(enu, anchor)
    return xyz2lla(xyz[:, 0], xyz[:, 1], xyz[:, 2])


def lla2enu(lla: np.ndarray, anchor):
    '''
    xyz2enu \circ lla2xyz
    lla should be (N, 3) 2d array, in (degrees, degrees, meters)
    '''
    x, y, z = lla2xyz(lla[:, 0], lla[:, 1], lla[:, 2])
    return xyz2enu(np.hstack((x, y, z)), anchor)

