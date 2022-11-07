import sys
import os
import itertools as it
from functools import reduce


def _compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
    '''
    compose(f, g, h) => f \circ g \circ h
    '''
    return reduce(_compose2, fs)


def bind1st(mf, *params, **kw):
    '''
    bind1st(f, a, b) => lambda x: f(x, a, b)
    bind member function: bind1st(str.split, ', ')
    '''
    def f(o):
        return mf(o, *params, **kw)
    return f


class seq:
    __slots__ = "data"

    def __init__(self, sequence):
        self.data = sequence

    def foreach(self, f):
        def doit(e):
            f(e)
            return e

        return self.map(doit)

    def map(self, f):
        return seq(map(f, self.data))

    def filter(self, f):
        return seq(filter(f, self.data))

    def fold(self, f, v):
        return reduce(f, self.data, v)

    def reduce(self, f):
        return reduce(f, self.data)

    def count(self, f=None):
        def positive(x):
            return 1 if (f is None or f(x)) else 0
        return sum(map(positive, self.data))

    def first(self, f=None):
        for d in self.data:
            if f is None or f(d):
                return d
        return None

    def to_list(self):
        return list(self.data)

    def __iter__(self):
        return self.data.__iter__()


def enum_lines(file: str):
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def enum_all_files(path):
    '''
    yield all files in path, return absolute path to use.
    '''
    for fld, dirs, files in os.walk(path):
        for f in files:
            yield os.path.join(fld, f)


def first_of(f, seq):
    return next(filter(f, seq), None)


def any_of(f, seq):
    return first_of(f, seq) is not None


def count_if(f, seq):
    return sum(map(f, seq))


def min_by(f, seq):
    minele, minval = None, None
    for e in seq:
        v = f(e)
        if minval is None or minval > v:
            minele, minval = e, v

    return minele, minval


def unique_match(src, dst, mfun):
    '''
    @param mfun: (s, d) -> score
    @return index pairs: [i, j, value]
    '''
    matches = {}  # j: (i, distance)
    for i, s in enumerate(src):
        j, value = min_by(lambda i: mfun(s, dst[i]), range(len(dst)))
        if (not j in matches) or value < matches[j][1]:
            matches[j] = (i, value)

    return [(i, j, dis) for j, (i, dis) in matches.items()]


def identity(x): return x


def use_logging(logfile: str = None):
    import logging
    if logfile is None:
        logging.basicConfig(
            level=logging.DEBUG, format='%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(logfile=logfile, filemode='w',
                            level=logging.DEBUG, format='%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s')
