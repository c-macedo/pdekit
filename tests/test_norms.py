import numpy as np
import pytest

from pdekit.norms import norm_inf, norm_l2


def test_norm_inf_absolute_value():
    assert norm_inf([-1, 2, -5]) == 5

def test_norm_inf_plain():
    assert norm_inf([1, 2, -5]) == 5

def test_norm_inf_zero():
    assert norm_inf([0, 0, 0]) == 0

def test_norm_inf_empty():
    with pytest.raises(ValueError):
        norm_inf([])

def test_norm_inf_2d():
    assert norm_inf([[1, 2, 3], [4, 10, 6], [7, 8, 9]]) == 10

def test_norm_l2_basic():
    assert norm_l2([3, 4], 0.25) == pytest.approx(2.5)

def test_norm_l2_resolution_independence_1d():
    assert norm_l2(np.ones(10), 1/10) == pytest.approx(norm_l2(np.ones(1000), 1/1000))

@pytest.mark.parametrize("nx, ny", [(4, 4), (40, 20)])
def test_norm_l2_resolution_independence_2d(nx, ny):
    dx, dy = 1/nx, 2/ny
    h = (dx, dy)
    assert norm_l2(np.ones(nx*ny).reshape(nx, ny), h) == pytest.approx(np.sqrt(2))

def test_norm_l2_scaling():
    assert norm_l2([-3, -6, -9, -12], 1/4) == pytest.approx(3 * norm_l2([1, 2, 3, 4], 1/4))

def test_norm_l2_empty():
    with pytest.raises(ValueError):
        norm_l2([], 1)

def test_norm_l2_1d_incorrect_hdim():
    with pytest.raises(ValueError, match="dimension"):
        norm_l2([1, 2, 3], (0.1, 0.1))

def test_norm_l2_2d_incorrect_hdim():
    with pytest.raises(ValueError, match="dimension"):
        norm_l2([[1, 2, 3], [4, 5, 6]], (0.1,))

def test_norm_l2_2d_correct_hdim():
    assert norm_l2([[1, 2], [2, 4]], (0.4, 0.1)) == pytest.approx(1)

def test_norm_l2_2d():
    assert norm_l2([[1, 1, 1], [1, 1, 1], [1, 1, 1]], (1/3, 1/3)) == pytest.approx(1)

@pytest.mark.parametrize("n", [10, 100, 1000])
def test_norm_l2_conv_to_cont_l2_norm(n):
    mesh = np.linspace(0, 1, n+1)
    h = 1/n
    assert norm_l2(np.sin(np.pi * mesh), h) == pytest.approx(np.sqrt(1/2))

def test_norm_l2_negative_weight():
    with pytest.raises(ValueError, match="positive"):
        norm_l2([[1, 2, 3], [4, 5, 6]], (0.1, -0.1))

def test_norm_l2_NaN():
    with pytest.raises(ValueError, match="positive"):
        norm_l2([[1, 2, 3], [4, 5, 6]], (0.1, float('nan')))

