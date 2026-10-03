"""This module holds the L-2 and L-infinity norms for vectors
|e|_2 = (h * sum_i(e_i^2))^(1/2), |e|_infinity = max_i |e_i|"""
import numpy as np


def norm_inf(arr):
    """Return the L-infinity norm on arrays of any shape
    Raise ValueError for empty arrays"""
    arr = np.asarray(arr)
    return np.max(np.abs(arr))

def norm_l2(arr, cell_volume):
    """Return L-2 norm on arrays of any shape, Cell volume is h in 1D and dx * dy in 2D
    Raise ValueError for empty arrays and invalid cell volume"""
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Empty matrix given")
    if cell_volume <= 0:
        raise ValueError("Invalid cell volume given")
    return np.sqrt(cell_volume * np.sum(arr**2))