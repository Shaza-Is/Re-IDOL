import numpy as np


def q_conjugate(quaternion: np.ndarray):
    quaternion[1:] = -quaternion[1:]

def qv_mult(quaternion: np.ndarray):
    quaternion2 = (0.0, ) + tuple(vector)
    quaternion2 = np.array(quaternion2)
    
    mult_result1 = q_mult(quaternion1, quaternion2)
    mult_result2 = q_mult(mult_result1, q_conjugate(quaternion1))
    
    return mult_result2[1:]

def q_mult(quaternion1: np.ndarray, quaternion2: np.ndarray):
    w1, x1, y1, z1 = tuple(quaternion1)
    w2, x2, y2, z2 = tuple(quaternion2)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return np.array([w, x, y, z])

def apply_rotations_to_matrix():
    for index, points in enumerate(coord_matrix):
        quaternion = quaternion_matrix[index]
        points_after_rotation = qv_mult(quaternion, points)
        coord_matrix[index] = points_after_rotation
    
    return coord_matrix

