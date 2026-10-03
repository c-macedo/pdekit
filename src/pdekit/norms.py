"""This doc holds the L-2 and L-infinity norms for vectors
|e|_2 = Euclidean norm, |e|_infinity = max_i |e_i|"""
import numpy as np


def norm_inf(arr):
    """Return the L-infinity norm of array"""
    np.asarray(arr)
    return np.max(np.abs(arr))

