"""Test cases for numerical solvers."""
import unittest
import math
import numpy as np
from numerical_solvers import rk4, adams_bashforth, forward_euler

def diffeq(y, t):
    [u] = y
    dudt = -100 * u + 100 * math.sin(t)

    return dudt

class TestSolvers(unittest.TestCase):

    def test_RK4(self):
        u0 = [0]
        t1 = np.linspace(0, 3, 121)

        u1 = rk4(diffeq, u0, t1)
        self.assertAlmostEqual(float(u1[-1]), 0.151, places=3)

    def test_adams_bathforth(self):
        u0 = [0]
        t1 = np.linspace(0, 3, 5000)

        u1 = adams_bashforth(diffeq, u0, t1)
        self.assertAlmostEqual(float(u1[-1]), 0.151, places=3)

    def test_forward_euler(self):
        u0 = [0]
        t1 = np.linspace(0, 3, 500)

        u1 = forward_euler(diffeq, u0, t1)
        self.assertAlmostEqual(float(u1[-1]), 0.151, places=3)

if __name__ == '__main__':
    unittest.main()

