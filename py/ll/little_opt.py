from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from typing import List, Callable, Literal
import logging
from texttable import Texttable

Loss = Callable[[float], float]


def make_huber_loss(a):
    b = a ** 2
    def huber(s): return 1 if s <= b else (2 * a * s ** 0.5 - b) / s
    return huber


class Param:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.bound: np.ndarray | None = None  # (dim, 2)
        self.fixed: bool = False
        self._b: int = -1
        self._e: int = -1

    @property
    def dim(self): return len(self.x)

    def __str__(self): return f"Dim[{self.dim}]"
    def __repr__(self): return self.__str__()


class Fun:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.params: List[Param]
        self.loss: Loss | None = None
        self.W: np.ndarray | None = None
        self._b: int
        self.tag: str = ''

    def f(self, *xs):
        r = self.f0(*xs)
        if self.loss is not None:
            r = self.loss(r @ r) * r
        if self.W is not None:
            r = self.W @ r
        return r

    def f0(self, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def f1(self, *xs) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self):
        xd = ','.join(map(str, (x.dim for x in self.params)))
        return f"Fun[{xd}->{self.dim}]"

    def __repr__(self): return self.__str__()


class LittleProblem:
    def __init__(self) -> None:
        self.params: List[Param] = []
        self.funs: List[Fun] = []
        self._ydim: int
        self._jac_sparse = None  # optional csr_array

    def add_param(self, x: np.ndarray | Param):
        if isinstance(x, np.ndarray):
            x = Param(x)
        self.params.append(x)
        return self.params[-1]

    def add_fun(self, fun: Fun):
        self.funs.append(fun)
        return fun

    def add_simple_fun(self, dim: int, fn: Callable):
        f = Fun(dim)
        f.f0 = fn
        return self.add_fun(f)

    def solve(self, *, sparse=False, puts=logging.debug, method: Literal['trf', 'dogbox', 'lm'] = 'trf'):
        self._puts = puts
        self._build(sparse, puts)
        x0 = np.hstack([p.x for p in self.params if not p.fixed])

        if any(x.bound is not None for x in self.params):
            def auto_bound(x: Param):
                return x.bound if x.bound is not None else np.tile(np.r_[-np.inf, np.inf], (x.dim, 1))
            bounds = np.vstack([auto_bound(x) for x in self.params]).T
            self._puts('bound enabled.')
            # print(bounds)
        else:
            bounds = (-np.inf, np.inf)

        puts('before:\n' + self._hist_costs())

        rt = least_squares(self._cost_fn, x0, bounds=bounds,
                           jac_sparsity=self._jac_sparse, method=method)
        for p in self.params:
            if not p.fixed:
                p.x = rt.x[p._b: (p._b + p.dim)]

        puts('after:\n' + self._hist_costs())

        return rt

    def _build(self, sparse: bool, puts):
        n = 0
        for p in self.params:
            if p.fixed:
                continue
            p._b = n
            p._e = n + p.dim
            n += p.dim

        cnt = sum(p.dim for p in self.params)
        puts(f'params constructed: {cnt}-> {n}.')

        # remove unused function
        def not_contributing(f: Fun):
            assert f.params, "fun with empty params, possibly uninitialized."
            return all(p.fixed for p in f.params)
        cnt = len(self.funs)
        self.funs = [f for f in self.funs if not not_contributing(f)]
        if cnt != len(self.funs):
            puts(f'{cnt- len(self.funs)} fixed funs removed.')

        m = 0
        for f in self.funs:
            f._b = m
            m += f.dim
        self._ydim = m

        self._puts(f'problem built: {n} |-> {m}')

        # build sparse
        if sparse:
            ri, ci = [], []
            for f in self.funs:
                for p in f.params:
                    if not p.fixed:
                        for r in range(f._b, f._b + f.dim):
                            ci.extend(range(p._b, p._b + p.dim))
                            ri.extend([r, ] * p.dim)

            v = np.ones_like(ri)
            self._jac_sparse = csr_array((v, (ri, ci)), (m, n))
            self._puts(
                f'sparse structure built: {len(ri)}/{m* n} = {len(ri)/(m* n) * 100:.2f}%')

            # print("jac: ", self._jac_sparse.toarray())
        else:
            self._jac_sparse = None

    def _hist_costs(self) -> str:
        counts = defaultdict(int)
        costs = defaultdict(float)
        errors = defaultdict(float)
        for f in self.funs:
            counts[f.tag] += 1

            r = f.f(*[p.x for p in f.params])
            costs[f.tag] += np.atleast_1d(r) @ np.atleast_1d(r)

            r = f.f0(*[p.x for p in f.params])
            errors[f.tag] += np.atleast_1d(r) @ np.atleast_1d(r)

        header = ['tag', 'count', 'total-res',
                  'mean-res', 'total-err', 'mean-err']
        vals = []

        for t in counts.keys():
            n, r, e = counts[t], costs[t], errors[t]
            vals.append([t, n, r, r/n, e, e/n])

        tt = Texttable()
        tt.set_deco(Texttable.BORDER | Texttable.HEADER)
        tt.add_rows([header])
        tt.add_rows(vals, header=False)
        return tt.draw()

    def _cost_fn(self, x):
        y = np.zeros(self._ydim)
        for f in self.funs:
            xs = (p.x if p.fixed else x[p._b: p._e] for p in f.params)
            y[f._b: (f._b + f.dim)] = f.f(*xs)
        return y

    def __str__(
        self): return f"LittleProblem[{len(self.params)} params, {len(self.funs)} funs]"

    def __repr__(self): return self.__str__()


if __name__ == '__main__':
    from ll import use_logging

    use_logging()

    def test_simple():
        logging.info('test simple')
        lp = LittleProblem()

        x0 = lp.add_param(np.r_[0])
        x1 = lp.add_param(np.r_[1])
        x2 = lp.add_param(np.r_[2])
        # x3.fixed = True

        def eq(x, y): return 2*(x - y)

        def add_simple_obs(p, v, w=1.):
            # return lp.add_simple_fun(1, lambda x: x-v).set_params(p)
            f = lp.add_simple_fun(1, lambda x: x - v)
            f.params = [p, ]
            f.W = np.atleast_2d(w)
            return f

        add_simple_obs(x0, 1)
        add_simple_obs(x1, 1.1)
        add_simple_obs(x2, 0.8)
        add_simple_obs(x2, 10).loss = make_huber_loss(1)

        lp.add_simple_fun(1, eq).params = [x0, x1]
        lp.add_simple_fun(1, eq).params = [x1, x2]
        lp.add_simple_fun(1, eq).params = [x0, x2]

        print(lp)
        r = lp.solve(sparse=True)
        # print(r)
        print(x0.x, x1.x, x2.x)

    def test_2d_transform():
        logging.info('test 2d transform')

        import math

        class TransfromedDiff(Fun):
            def __init__(self, src: np.ndarray, dst: np.ndarray) -> None:
                super().__init__(2)
                self._src, self._dst = src, dst

            def f0(self, x) -> np.ndarray:
                c, s = math.cos(x[2]), math.sin(x[2])
                return np.c_[[c, s], [-s, c]] @ self._src + x[:2] - self._dst

        lp = LittleProblem()

        x = lp.add_param(np.zeros(3))
        da = math.radians(30)
        x.bound = np.c_[[-np.inf, -np.inf, -da], [np.inf, np.inf, da]]

        lp.add_fun(TransfromedDiff(np.r_[0, 0], np.r_[0, 0])).params = [x, ]
        lp.add_fun(TransfromedDiff(np.r_[1, 0], np.r_[1, 1])).params = [x, ]

        r = lp.solve()

        print(lp, lp.params, lp.funs)
        print(r)
        x, y, theta = x.x
        print(x, y, math.degrees(theta))

    test_simple()
    test_2d_transform()
