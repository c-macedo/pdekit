"""This module holds the L-2 and L-infinity norms for vectors
|e|_2 = (h * sum_i(e_i^2))^(1/2), |e|_infinity = max_i |e_i|"""

import numpy as np


def norm_inf(arr):
    """Return the L-infinity norm on arrays of any shape
    Raise ValueError for empty arrays"""

    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("empty matrix given")
    return np.max(np.abs(arr))


def norm_l2(arr, h):
    """Return L-2 norm on arrays of any shape, Cell volume is h in 1D and dx * dy
    in 2D Raise ValueError for empty arrays and invalid cell volume"""

    arr = np.asarray(arr)
    weights = np.atleast_1d(h)

    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")
    if weights.size != arr.ndim:
        raise ValueError(
            f"tuple size ({weights.size}) doesn't match array dimension ({arr.ndim})"
        )
    if arr.size == 0:
        raise ValueError("empty matrix given")
    if np.all(weights > 0):
        return np.sqrt(np.prod(weights) * np.sum(arr**2))
    else:
        raise ValueError(f"expected positive weights, got: {h}")
