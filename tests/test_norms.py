from pdekit.norms import norm_inf
import numpy as np

def test_norm_inf():
    assert norm_inf(np.array([-1, 2, -5])) == 5

def test_norm_inf_plain():
    assert norm_inf([1, 2, -5]) == 5
