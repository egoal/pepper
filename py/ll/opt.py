import numpy as np
import scipy.linalg as la
import math


class Var:
    def __init__(self, id: int):
        self._id = id
        self._fixed = None
        # handle by Pro
        self._x, self._new_x = None, None
        self._s, self._e = -1, -1  # mark of range

    def id(self) -> int:
        return self._id

    def x(self) -> np.ndarray:
        return self._x

    def dim(self) -> int:
        return self._x.shape[0]

    def plus(self, x: np.ndarray, dx: np.ndarray):
        return x + dx

    def fix(self):
        self._fixed = True


class Fun:
    def __init__(self) -> None:
        # handle by optimizer
        self._params = []
        self._s, self._e = -1, -1  # mark of range

    def dim(self) -> int:
        raise NotImplemented

    def f0(self, *params) -> np.ndarray:
        raise NotImplemented

    def f1(self, *params) -> np.ndarray:
        return NotImplemented


class Pro:
    def __init__(self) -> None:
        self._vars = []
        self._funs = []

        self.initial_radius = 1e3
        self.max_iterations = 10
        self.max_inner_iterations = 10

    def new_var(self, init: np.ndarray) -> Var:
        v = Var(len(self._vars))
        v._x = init.astype(np.float)
        self._vars.append(v)
        return v

    def new_fun(self, fun: Fun, *params):
        fun._params = params
        self._funs.append(fun)
        return fun

    def solve(self):
        self._set_ranges()

        xdim, ydim = self._vars[-1]._e, self._funs[-1]._e
        J, e = np.zeros((ydim, xdim)), np.zeros(ydim)
        print(f'J, e allocated: {J.shape}, {e.shape}')

        mu, v = self.initial_radius, 2

        # main loop
        for it in range(self.max_iterations):
            for f in self._funs:
                f0 = f.f0(*[p.x() for p in f._params])
                f1 = f.f1(*[p.x() for p in f._params])

                e[f._s: f._e] = f0
                sid = 0
                for p in f._params:
                    eid = sid + p.dim()
                    if not p._fixed:  # no gradiant for fixed var
                        J[f._s: f._e, p._s: p._e] = f1[sid: eid]
                    sid = eid

            cost, JtJ, Jte = e.T @ e, J.T @ J, J.T @ e

            if Jte.T @ Jte < 1e-12:
                print('converged.')
                break

            # inner loop here
            nit = 0
            while nit < self.max_inner_iterations:
                dx = la.solve(JtJ + np.identity(xdim) *
                              (1/mu), -Jte, assume_a='sym')

                for p in self._vars:
                    p._new_x = p.plus(p._x, dx[p._s: p._e])

                newcost = 0
                for f in self._funs:
                    f0 = f.f0(*[p._new_x for p in f._params])
                    newcost += f0.T @ f0
                error_cost_change = (cost - newcost) * .5

                if error_cost_change > 0:  # step acceptable
                    model_cost_change = J @ dx
                    model_cost_change = - \
                        model_cost_change.T @ (e + model_cost_change*.5)

                    rho = error_cost_change / model_cost_change

                    mu /= max(1/3, 1 - (2*rho-1)**3)
                    v = 2
                    for p in self._vars:
                        if not p._fixed:
                            p._x = p._new_x  # accept
                    break
                else:
                    mu /= v
                    v *= 2

                ++nit

            print(f'{it}| cost: {cost:.6}, mu: {mu}, nit: {nit}')

    def _set_ranges(self):
        s = 0
        for p in self._vars:
            e = s + p.dim()
            p._s, p._e = s, e
            s = e

        s = 0
        for f in self._funs:
            e = s + f.dim()
            f._s, f._e = s, e
            s = e

# test


class PointDiff(Fun):
    def dim(self) -> int:
        return self._params[0].dim()

    def f0(self, *params) -> np.ndarray:
        return params[0] - params[1]

    def f1(self, *params) -> np.ndarray:
        return np.identity(self.dim()), -np.identity(self.dim())


def test_simple_quadratic():
    '''
    ||x_1 - 1, x_2 - 2, x_1 - x_2|| with x_0 = [0, 0]
    '''
    p = Pro()
    x1, x2 = p.new_var(np.r_[0]), p.new_var(np.r_[0])
    c1, c2 = p.new_var(np.r_[1]), p.new_var(np.r_[2])
    c1.fix()
    c2.fix()
    f1 = p.new_fun(PointDiff(), x1, c1)
    f2 = p.new_fun(PointDiff(), x2, c2)
    f3 = p.new_fun(PointDiff(), x1, x2)

    p.solve()

    print(x1.x(), x2.x(), c1.x(), c2.x())


# (x, y, theta)
class TransfromedDiff(Fun):
    def __init__(self, src: np.ndarray, dst: np.ndarray) -> None:
        super().__init__()
        self._src, self._dst = src, dst

    def dim(self) -> int:
        return 2

    def f0(self, *params) -> np.ndarray:
        p = params[0]
        c, s = math.cos(p[2]), math.sin(p[2])
        return np.c_[[c, -s], [c, s]] @ self._src + p[:2] - self._dst

    def f1(self, *params) -> np.ndarray:
        p = params[0]
        c, s = math.cos(p[2]), math.sin(p[2])
        j = np.zeros((2, 3))
        j[:, :2] = np.identity(2)
        j[0, 2] = -self._src[0] * s + self._src[1] * c
        j[1, 2] = -self._src[0] * c - self._src[1] * s
        return j


def test_2d_transform():
    '''
    || dst - T* src ||
    '''
    p = Pro()
    x = p.new_var(np.zeros(3))
    f1 = p.new_fun(TransfromedDiff(np.r_[0, 0], np.r_[0, 0]), x)
    f2 = p.new_fun(TransfromedDiff(np.r_[1, 1], np.r_[1, 0]), x)

    p.solve()

    x = x.x()
    print(x[:2], math.degrees(x[2]))


if __name__ == '__main__':
    print('---')
    test_simple_quadratic()
    print('---')
    test_2d_transform()
