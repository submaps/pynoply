import unittest
import numpy as np
from krig.krig_imp import plot_grid, OK, SK


class TestOpt(unittest.TestCase):
    def test_opt(self):
        np.random.seed(0)
        grid_init = np.zeros((100, 100), dtype='float32')  # float32 gives us a lot precision
        x, y = np.random.randint(0, 100, 10), np.random.randint(0, 100, 10)  # CREATE POINT SET.
        v = np.random.randint(0, 10, 10)  # THIS IS MY VARIABLE
        grid_ok = OK(x, y, v, (50, 30), grid_init)
        plot_grid(x, y, v, grid_ok)
        grid_sk = SK(x, y, v, (50, 30), grid_init)
        plot_grid(x, y, v, grid_sk)
        mean_diff = (grid_ok - grid_sk).mean()
        self.assertTrue(mean_diff < 0.1)

    def test_opt2(self):
        np.random.seed(0)
        grid_init = np.zeros((100, 100), dtype='float32')  # float32 gives us a lot precision
        x, y = np.random.randint(0, 100, 10), np.random.randint(0, 100, 10)  # CREATE POINT SET.
        v = np.random.randint(0, 10, 10)  # THIS IS MY VARIABLE
        # optimal kriging
        print(v)
        print(x, y)
