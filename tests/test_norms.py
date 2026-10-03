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

def test_norm_l2_resolution_independence():
    assert norm_l2(np.ones(10), 1/10) == pytest.approx(norm_l2(np.ones(1000), 1/1000))

def test_norm_l2_scaling():
    assert norm_l2([-3, -6, -9, -12], 1/4) == pytest.approx(abs(-3) * norm_l2([1, 2, 3, 4], 1/4))

def test_norm_l2_empty():
    with pytest.raises(ValueError):
        norm_l2([], 1)

def test_norm_l2_2d():
    assert norm_l2([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == pytest.approx(1)

@pytest.mark.parametrize("n", [10, 100, 1000])
def test_norm_l2_conv_to_cont_l2_norm(n):
    mesh = np.linspace(0, 1, n+1)
    h = 1/n
    assert norm_l2(np.sin(np.pi * mesh), h) == pytest.approx(np.sqrt(1/2))
