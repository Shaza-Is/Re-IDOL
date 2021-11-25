import pytest
import numpy as np

from app.utils.quaternions import q_conjugate, q_mult
from pprint import pprint

def test_qmult():
    vector1 = np.array([0.2, 0.5, 0.1, 0.7])
    vector2 = np.array([0.3, 0.65, 0.28, 0.89])

    result = q_mult(vector1, vector2)
    expected = np.array([-0.916, 0.173, 0.096, 0.463])

    assert (result == expected).all() == True


@pytest.mark.skip("Not implemented yet")
def test_qvmult():
    pass

def test_conjugate():
    random_vector = np.random.rand(4)
    result = q_conjugate(random_vector)
    expected = random_vector
    expected[1:] = -expected[1:]

    assert (random_vector == expected).all() == True

@pytest.mark.skip("Not implemented yet")
def test_apply_matrix_rotations():
    pass