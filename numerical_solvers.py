"""Integration schemes."""
import numpy as np
import math
import unittest

def forward_euler(func, y0, t):

    y = np.full((len(t), len(y0)), None)

    y[0, :] = y0

    for index in range(1, len(t)):
        delta_t = t[index] - t[index - 1]
        y[index, :] = y[index - 1, :] + np.array(func(y[index - 1, :], t[index - 1])) * delta_t

    return y

def rk4(func, y0, t):

    y = np.full((len(t), len(y0)), None)

    y[0, :] = y0

    for index in range(1, len(t)):
        delta_t = t[index] - t[index - 1]

        k1 = np.array(func(y[index - 1, :], t[index - 1])) * (1 / 2)
        k2 = np.array(func(y[index - 1, :] + delta_t * k1, t[index - 1] + delta_t / 2)) * (1 / 2)
        k3 = np.array(func(y[index - 1, :] + delta_t * k2, t[index - 1] + delta_t / 2)) * (1 / 2)
        k4 = np.array(func(y[index - 1, :] + 2 * delta_t * k3, t[index])) * (1 / 2)

        y[index, :]  = (1 / 3) * (k1 + 2 * k2 + 2 * k3 + k4) * delta_t + y[index - 1, :]

    return y

def adams_bashforth(func, y0, t):

    y_init = rk4(func, y0, t[0:4])

    y = np.full((len(t), len(y0)), None)

    y[0:4, :] = y_init

    for index in range(4, len(t)): 
        delta_t = t[index] - t[index - 1]

        b1 = 55 / 24
        b2 = -59 / 24
        b3 = 37 / 24
        b4 = -9 / 24

        y[index, :] = delta_t * (b1 * np.array(func(y[index - 1, :], t[index - 1])) \
                               + b2 * np.array(func(y[index - 2, :], t[index - 2])) \
                               + b3 * np.array(func(y[index - 3, :], t[index - 3])) \
                               + b4 * np.array(func(y[index - 4, :], t[index - 4]))) \
                               + y[index - 1, :]

    return y


def main():
    
    def linear(y, t):
        x1, x2 = y
        dydt = [2, 3]

        return dydt

    y0 = [0, 0]
    t = np.linspace(0, 5, 51)

    sol = forward_euler(linear, y0, t)

    def diffeq(y, t):
        u = y
        dudt = -100 * u + 100 * math.sin(t)

        return dudt

if __name__ == '__main__':
    main()
