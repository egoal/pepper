#!/usr/bin/python3
'''
steepest descent and newton algorithm using the backtracking line search, 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
alpha: step size, 
rho: step size shrink ratio, 
c: sufficient check ratio
method: way to choose search direction, 
    gradient: descent direction
    newton: 
'''


def backtracking_line_search(f, g, h, x0, alpha, rho, c, method='gradient'):
    def descent_direction(x):
        p = -g(x)
        return p / np.linalg.norm(p)

    def newton_direction(x):
        p = -np.linalg.inv(h(x)).dot(g)
        return p / np.linalg.norm(p)

    methods = {
        'gradient': descent_direction,
        'newton': newton_direction,
    }

    assert (alpha > 0 and rho > 0 and rho < 1 and c > 0 and c < 1
            and method in methods)

    calc_direction = methods[method]

    a = alpha

    xs, fs = [], []
    xs.append(x0)
    fs.append(f(x0))
    for i in range(10):
        print('i: {}, f: {}, x: {}, '.format(
            i,
            fs[-1],
            xs[-1],
        ))

        x = xs[-1]

        p = calc_direction(x)
        while True:
            nf, lf = f(x + a * p), f(x) + c * a * g(x).dot(p)
            if nf > lf:
                # decrease, sufficient, then update
                xs.append(x + a * p)
                fs.append(nf)
                break
            else:
                # decrease step length
                a = a * rho

    print('i: -1, f: {}, x: {}, '.format(
        fs[-1],
        xs[-1],
    ))

    return xs, fs


if __name__ == '__main__':
    from test_functions import *

    x0 = np.r_[1.2, 1.2]
    xs, fs = backtracking_line_search(
        rosenbrock_2,
        rosenbrock_2_first,
        None,
        x0,
        1.,
        .8,
        .5,
        'gradient',
    )

    # plots
    plt.figure()
    plt.plot(fs)

    plt.figure()
    x, y = np.linspace(0, 2, 20), np.linspace(0, 2, 20)
    xx, yy = np.meshgrid(x, y)
    zz = rosenbrock_2_p(xx, yy)

    plt.contourf(xx, yy, zz, 20, alpha=0.75)
    c = plt.contour(xx, yy, zz, 20, colors='black')
    plt.clabel(c, inline=True, fontsize=10)

    plt.plot(x0[0], x0[1], 'ro')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, zz, )

    plt.show()
