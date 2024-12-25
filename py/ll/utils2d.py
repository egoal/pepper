from typing import Tuple
import numpy as np
import scipy.linalg as la
from . import ll
from . import calc
from functools import reduce


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


def range_valid(r): return r[0] <= r[1]


def range_intersect(r1, r2): return (max(r1[0], r2[0]), min(r1[1], r2[1]))


def overlap_circle(center, radius, s, e) -> Tuple[float, float]:
    dis, t = distance_to(center, s, e)
    if abs(dis) > radius or t < 0 or t > 1:
        return (np.inf, -np.inf)

    l = la.norm(e - s)
    dt = (radius ** 2 - dis ** 2) ** .5 / l

    return (t - dt, t + dt)


def overlap_line(s1, e1, s2, e2) -> Tuple[float, float]:
    se1 = e1 - s1
    ds, de = np.cross(se1, s2 - s1), np.cross(se1, e2 - s1)

    if abs(de - ds) < 1e-6:
        if de >= 0.:
            return (-np.inf, np.inf)
        else:
            return (np.inf, -np.inf)
    t = (0 - ds) / (de - ds)

    if ds < de:
        return (t, np.inf)
    else:
        return (-np.inf, t)


def overlap_convex(polygon, s, e) -> Tuple[float, float]:
    # todo: can be optimized.
    return reduce(range_intersect, map(lambda x: overlap_line(x[0], x[1], s, e), ll.adjacent(polygon)))


def overlap_capsule(s, e, radius, p1, p2):
    t1 = overlap_circle(s, radius, p1, p2)

    n = calc.normalized(e - s)
    pn = np.array([-n[1], n[0]]) * radius

    rect = np.array([s - pn, e - pn, e + pn, s + pn, s - pn])

    t2 = overlap_convex(rect, p1, p2)
    t3 = overlap_circle(e, radius, p1, p2)

    def __union(r1, r2): return (min(r1[0], r2[0]), max(r1[1], r2[1]))

    return reduce(__union, [t1, t2, t3])

