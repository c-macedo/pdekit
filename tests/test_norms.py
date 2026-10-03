import numpy as np
import pytest
from pdekit.norms import norm_inf


def test_norm_inf_absolute_value():
    assert norm_inf(np.array([-1, 2, -5])) == 5

def test_norm_inf_plain():
    assert norm_inf([1, 2, -5]) == 5

def test_norm_inf_zero():
    assert norm_inf([0, 0, 0]) == 0

def test_norm_inf_empty():
    with pytest.raises(ValueError):
        norm_inf(np.array([]))

def test_norm_inf_2D():
    assert norm_inf(np.array([[1, 2, 3], [4, 10, 6], [7, 8, 9]])) == 10