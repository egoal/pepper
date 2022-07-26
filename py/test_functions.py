import numpy as np
'''
rosenbrock function
'''


def rosenbrock_2(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_2_first(x):
    a = x[1] - x[0]**2
    return np.r_[-400 * a * x[0] - 2 * (1 - x[0]), 200 * a]


def rosenbrock_2_second(x):
    return np._c[[400 * (3 * x[0]**2 - x[1]) +
                  2, -400 * x[0]], [-400 * x, 200]]


def rosenbrock_2_p(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def rosenbrock_n(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.)**2. + (1 - x[:-1])**2.)
