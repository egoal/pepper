import logging
import os
import itertools as it
from functools import cache, partial, reduce
from typing import Any, Callable, Generator, Iterable, List, Tuple, TypeVar, Dict
from types import FunctionType

T, U = TypeVar('T'), TypeVar('U')


def never_reach(s=None): raise RuntimeError(s or "never reach")


def _compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
    '''
    compose(f, g, h) => f . g . h
    '''
    return reduce(_compose2, fs)


def partial_tail(mf, *params, **kw):
    '''
    return f(x, ...)

    1. member function: partial_tail(str.split, ', ')  ('a, b, c')
    2. no keywords: partial_tail(it.contains, 1) ([1, 2, 3])
    '''
    def f(o): return mf(o, *params, **kw)
    return f


def F(statement, params='x', tag='f'):
    '''simple eval function'''
    code = f"def {tag}({','.join(params)}): return {statement}"
    obj = compile(code, 'F', 'exec')
    return FunctionType(obj.co_consts[0], globals())


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


def first_of(f: Callable[[T], bool], seq: Iterable[T]) -> T | None:
    return next(filter(f, seq), None)


def flow(x, *fs): return reduce(lambda s, a: a(s), fs, x)


any_of = compose(any, map)
count_if = compose(sum, map)
sum_by = count_if


def each(fn, xs):
    for x in xs:
        if fn(x) == True:
            break


def min_by(f: Callable[[T], U | None], seq: Iterable[T]) -> Tuple[T | None, U | None]:
    '''
    f: x-> score?
    '''
    minele, minval = None, None
    for e in seq:
        v = f(e)
        if (v is not None) and (minval is None or v < minval):
            minele, minval = e, v

    return minele, minval


def max_by(f: Callable[[T], U | None], seq: Iterable[T]) -> Tuple[T | None, U | None]:
    '''
    f: x-> score?
    '''
    maxele, maxval = None, None
    for e in seq:
        v = f(e)
        if (v is not None) and (maxval is None or v > maxval):
            maxele, maxval = e, v

    return maxele, maxval


def unique_match(src: Iterable[T], dst: Iterable[U], fn: Callable[[T, U], float | None]) -> List[Tuple[T, U, float]]:
    '''
    fn: (s, d)-> cost?

    return [(s, d, cost)]
    '''
    matches = {}  # d: (s, cost)
    for s in src:
        d, cost = min_by(lambda x: fn(s, x), dst)
        if (cost is not None) and ((not d in matches) or cost < matches[d][1]):
            matches[d] = (s, cost)

    return [(s, d, cost) for d, (s, cost) in matches.items()]


def group_by(fn: Callable[[T], U], seq: Iterable[T]) -> Dict[U, List[T]]:
    '''
    for unsorted collections. implement with dict.
    '''
    from collections import defaultdict
    g = defaultdict(list)
    for x in seq:
        g[fn(x)].append(x)
    return g


def partition(fn: Callable[[T], bool], seq: Iterable[T]) -> Tuple[List[T], List[T]]:
    '''
    return (pos, neg)
    '''
    d = group_by(fn, seq)
    return d.get(True, []), d.get(False, [])


def identity(x): return x


def inverse(fn): return lambda *a, **kw: not fn(*a, **kw)


def not_none(x): return x is not None


memoize = cache


def flatten(xs):
    '''
    alias of it.chain

    consider full_flatten for deeper.
    '''
    return it.chain(*xs)


def full_flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from full_flatten(x, ignore_types)
        else:
            yield x


def take(n, seq): return it.islice(seq, n)


def every(n, xs):
    '''
    only ensured if n mod N
    '''
    i = iter(xs)
    while True:
        try:
            yield it.chain([next(i)], it.islice(i, n-1))
        except StopIteration:
            break


'''simple curry'''
def map_by(fn): return partial(map, fn)
def filter_by(fn): return partial(filter, fn)
def reduce_by(fn): return partial(reduce, fn)


def stared(fn): return lambda x: fn(*x)


def sided(fn): return lambda x: (x, fn(x))


def adjacent(seq):
    it = iter(seq)
    prev = next(it)
    for nxt in it:
        yield (prev, nxt)
        prev = nxt


def step_range(s, e, step):
    if step > 0:
        while s < e:
            s += step
            yield s - step
    else:
        while s > e:
            s += step
            yield s - step


def clamp(x, l, h): return max(l, min(h, x))


def use_logging(level=logging.DEBUG, logfile: str | None = None, logstd=True):
    handlers = []
    if logstd:
        handlers.append(logging.StreamHandler())

    if logfile:
        import os
        import time
        a, b = os.path.splitext(logfile)
        filename = a + time.strftime('%Y%m%d_%H-%M-%S') + b
        handlers.append(logging.FileHandler(filename))

        print('logfile created at:', os.path.realpath(filename))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
        handlers=handlers)


def directed_join(fn: Callable[[T, T], bool], xs: Iterable[T]) -> List[List[T]]:
    def add(s: List[List[T]], a: T):
        def get_index():
            for i, x in enumerate(s):
                if fn(a, x[0]):
                    return (i, True)
                elif fn(x[-1], a):
                    return (i, False)

        r = get_index()
        if r is None:
            s.append([a])
        elif r[1]:
            s[r[0]] = [a] + s[r[0]]
        else:
            s[r[0]].append(a)

        return s

    return reduce(add, xs, [])
