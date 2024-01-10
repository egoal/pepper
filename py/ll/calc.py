from typing import Callable, List
import numpy as np
import scipy.linalg as la
from scipy import sparse


def squared_norm(a: np.ndarray) -> float: return a @ a


def normalized(arr: np.ndarray) -> np.ndarray: return arr / la.norm(arr)


def upsample(line: np.ndarray, step: float) -> np.ndarray:
    '''
    this would not change orignal points
    @param line: (N, DIM)
    @return densed-line: (M, DIM)
    '''
    # size, _ = line.shape
    # re = []
    # for i in range(1, size):
    #     s, e = line[i-1, :], line[i, :]
    #     dp, dlen = e - s, la.norm(e - s)
    #     if dlen <= step:
    #         re.append(s)
    #     else:
    #         ds = dp / dlen * step
    #         i = 0
    #         while dlen> 1e-2:
    #             re.append(s+ ds* i)
    #             i += 1
    #             dlen -= step
    # re.append(line[-1, :])
    size, _ = line.shape
    re = []
    lappend = re.append
    lextend = re.extend
    algnorm = np.linalg.norm
    dps = line[1:, :] - line[:-1, :]
    dlens = algnorm(dps, axis=1)
    for i in range(1, size):
        s, dp, dlen = line[i - 1, :], dps[i - 1, :], dlens[i - 1]
        if dlen < step:
            lappend(s)
        else:
            n = int(dlen / step) + 1
            ds = (dp / dlen * step).reshape(-1, 1)
            dd = s.reshape(-1, 1) + ds * np.arange(n,
                                                   dtype=np.float64).reshape(1, -1)
            lextend(dd.T)
    lappend(line[-1, :])

    return np.array(re)


def downsample(line: np.ndarray, step: float) -> np.ndarray:
    '''
    this would not change orignal points
    @param line: (N, DIM)
    @return densed-line: (M, DIM)
    '''
    size, _ = line.shape
    re = [line[0, :]]
    acclen = 0.
    for i in range(1, size):
        acclen += la.norm(line[i, :] - line[i-1, :])
        if acclen > step:
            acclen = 0.
            re.append(line[i, :])

    if acclen > 0.:
        re.append(line[-1, :])

    return np.array(re)


def resample(line: np.ndarray, step: float) -> np.ndarray:
    ''' resample `line` with `step`, this does NOT ensure the original point

    all return points should be NEARLY evenly distributed, the last one excluded
    '''
    size, _ = line.shape
    re = [line[0, :]]
    leftlen = 0.
    dists = np.sum((line[1:, :] - line[:-1, :]) ** 2, axis=1) ** 0.5
    for i in range(1, size):
        # curlen = la.norm(line[i] - line[i-1])
        curlen = dists[i-1]
        if curlen == 0:
            continue
        curdir = (line[i] - line[i-1]) / curlen
        l = step - leftlen
        while l < curlen:
            re.append(line[i-1] + curdir * l)
            l += step

        leftlen = step - (l - curlen)

    if leftlen > 0.:
        re.append(line[-1])

    return np.array(re)


def length_of(line: np.ndarray):
    '''
    line: [N, Dim]
    '''
    return np.sum(np.sum((line[1:, :] - line[:-1, :]) ** 2, axis=1) ** 0.5)


def pca(points):
    rows, _ = points.shape
    cen = points.sum(axis=0) / rows

    nps = points - cen
    U, S, _ = la.svd(nps.T @ nps)

    return (cen, U, S)


def umeyama(src, dst, w=None):
    '''
    \sum |W(dst - (R* src+ t)|
    @param src, dst: (N, DIM)
    @return R, t
    '''
    N, DIM = src.shape
    w = np.ones((N, 1)) if w is None else w.reshape((-1, 1))
    assert N == dst.shape[0] and N == w.shape[0] and N >= DIM

    bar_src = np.sum(src * w, axis=0) / np.sum(w)
    bar_dst = np.sum(dst * w, axis=0) / np.sum(w)
    bar_src, bar_dst = bar_src.T, bar_dst.T

    A = (src - bar_src).T @ sparse.diags(w.reshape(-1)) @ (dst - bar_dst)
    U, s, Vt = la.svd(A)

    S = np.eye(DIM)
    if la.det(U) * la.det(Vt) < 0:
        S[DIM-1, DIM-1] = -1
    R = Vt.T @ S @ U.T
    t = bar_dst - R @ bar_src

    return R, t


def inverse_transform(T: np.ndarray):
    '''
    T [4, 4]
    '''
    r = np.identity(4)
    r[:3, :3] = T[:3, :3].T
    r[:3, -1] = - r[:3, :3] @ T[:3, -1]
    return r


def simple_cluster(N, fn: Callable[[int, int], bool]) -> List[List[int]]:
    '''implemnet by simple DENSE matrix.'''
    from scipy.sparse.csgraph import connected_components

    M = np.eye(N)
    for i in range(N):
        for j in range(i+1, N):
            if fn(i, j):
                M[i, j] = 1
    n, flags = connected_components(M)
    return [list(np.where(flags == i)[0]) for i in range(n)]
