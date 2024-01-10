from typing import Tuple
import numpy as np
import scipy.linalg as la
from . import ll


def distance_to(p, s, e) -> Tuple[float, float]:
    '''
    @return (dis, t): foot = lerp(s, e, t)
    '''
    se = e - s
    len2 = se @ se
    if len2 < 1e-6:
        return (0., la.norm(p - s))

    t = (p - s) @ se / len2
    dis = np.cross(se, p - s) / len2 ** 0.5

    return (dis, t)


def distance_to_line(p, line) -> Tuple[float, float]:
    '''
    @param line: (N, 2)
    @return (dis, t): [0, N-1]
    '''

    i: int = np.argmin(np.sum((line - p) ** 2, axis=1))

    lam, distance = i, la.norm(line[i, :] - p)

    if i > 0:
        dis, t = distance_to(p, line[i-1], line[i])
        if t > 0 and t < 1:
            distance = abs(dis)
            lam = i - 1 + t
    if i < line.shape[0] - 1:
        dis, t = distance_to(p, line[i], line[i + 1])
        if t > 0 and t < 1 and abs(dis) < distance:
            distance = abs(dis)
            lam = i + t

    return (distance, lam)


def foot_on_line(p, line) -> np.ndarray | None:
    '''
    @param line (N, 2)
    @return maybe (2,)
    '''
    _, lam = distance_to_line(p, line)
    if lam > 0 and lam < len(line)-1:
        i, d = int(lam), lam - int(lam)
        return line[i] * (1-d) + line[i+1] * d


def intersect_segment(s1, e1, s2, e2):
    '''
    lerp(s1, e1, t1) = lerp(s2, e2, t2)
    @return t1, t2: inf if parallel.
    '''
    se1, se2 = e1 - s1, e2 - s2
    a = np.cross(se2, se1)
    if abs(a) < 1e-6:
        return np.inf, np.inf

    s12 = s2 - s1
    t1 = np.cross(se2, s12) / a
    t2 = np.cross(-s12, se1) / a
    return (t1, t2)


def intersect_line(s, e, line: np.ndarray):
    '''
    @param line: (N, 2)
    @return [(t1, t2)] where t1 in [0, 1], t2 in [0, N-1]
    '''
    assert line.shape[0] > 1 and line.shape[1] == 2

    def intersect_at(i):
        t1, t2 = intersect_segment(s, e, line[i], line[i+1])

        if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
            return (t1, t2 + i)
        else:
            return None

    return list(filter(ll.not_none, map(intersect_at, range(line.shape[0]-1))))
