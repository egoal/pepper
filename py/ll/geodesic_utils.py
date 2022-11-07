import math
import numpy as np

RADIUS = 6378137.0      # semi-major axis
e = 1 / 298.257223563   # flattening
e1s = 0.00669437999013  # e1 sqaure
wie = 2.0 * math.pi / (23.0 * 3600.0 + 56.0 * 60.0 + 4.0)


def ll2Rne(lon, lat) -> np.ndarray:
    '''
    @return 3x3 rotation matrix
    '''
    clon, slon = math.cos(lon), math.sin(lon)
    clat, slat = math.cos(lat), math.sin(lat)

    return np.c_[[-slon, clon, 0], [-slat*clon, -slat*slon, clat], [clat*clon, clat*slon, slat]]


def lla2xyz(lla) -> np.ndarray:
    '''
    @param lla: Nx3 [lon, lat, alt] 
    @return Nx3 [x, y, z] in ecef
    '''
    lon, lat, alt = lla[:, 0], lla[:, 1], lla[:, 2]

    clon, slon = np.cos(lon), np.sin(lon)
    clat, slat = np.cos(lat), np.sin(lat)

    N = RADIUS/np.sqrt(clat**2 + (1-e)**2 * slat**2)
    h = N+alt
    return np.hstack((h * clat * clon, h * clat*slon, (N*(1-e)**2 + alt)*slat))


def xyz2enu(xyz, anchor) -> np.ndarray:
    '''
    @param xyz: Nx3
    @anchor 3 [lon, lat, alt]
    '''
    return (xyz - lla2xyz(anchor.reshape((-1, 3)))) @ (ll2Rne(anchor[0], anchor[1])).T


def lla2enu(lla, anchor):
    '''
    compose: xyz2enu \circ lla2xyz
    '''
    return xyz2enu(lla2xyz(lla), anchor)


def xyz2lla(xyz) -> np.ndarray:
    '''
    @param xyz: Nx3
    '''
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    L = np.arctan2(y, x)
    tmpB = z / np.sqrt(x**2 + y**2)
    B = np.arctan(tmpB/(1-e)**2)

    iter, max_iterations = 0, 500
    err, thresh = 1, 1e-15
    while iter < max_iterations and err > thresh:
        R1 = x / (np.cos(L) * np.cos(B))
        R2 = RADIUS / np.sqrt(1 - e1s * np.sin(B)**2)
        newB = np.arctan(tmpB * R1 / (R1-R2*e1s))
        err = max(abs(newB - B))
        B = newB

    return np.vstack((L, B, R1-R2)).T


def enu2xyz(enu, anchor) -> np.ndarray:
    '''
    @param enu: Nx3
    '''
    return enu @ ll2Rne(anchor[0], anchor[1]).T + lla2xyz(np.r_[anchor[0], anchor[1], 0.].reshape((-1, 3)))


def enu2lla(enu, anchor):
    return xyz2lla(enu2xyz(enu, anchor))

# wgs, gcj


def _transform_latitude(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * \
        lat + 0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
            np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
            np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(lat / 12.0 * np.pi) +
            320 * np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_longitude(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
        0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
            np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
            np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(lng / 12.0 * np.pi) +
            300.0 * np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def _transform(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = _transform_latitude(lng - 105.0, lat - 35.0)
    dlng = _transform_longitude(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    lat += dlat
    lng += dlng

    return lng, lat


def wgs2gcj(wgs):
    '''
    @param wgs: Nx2 radians
    '''
    gcj = np.degrees(wgs)
    gcj[:, 0], gcj[:, 1] = _transform(gcj[:, 0], gcj[:, 1])
    return np.radians(gcj)


def gcj2wgs(gcj):
    '''
    @param wgs: Nx2 radians
    '''
    wgs = np.degrees(gcj)
    wgs[:, 0], wgs[:, 1] = _transform(wgs[:, 0], wgs[:, 1])
    wgs = np.radians(wgs)
    return gcj*2 - wgs
