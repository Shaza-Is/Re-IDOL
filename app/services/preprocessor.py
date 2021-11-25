import pandas as pd
import numpy as np
import tensorflow as tf

from typing import Tuple
import tensorflow
from tensorflow.keras import Input
from tensorflow import Tensor

from app.core.config import REORIENT_NET_LOOKBACK, REORIENT_NET_BATCH_SIZE
# from app.resources.constants import REORIENT_FEATURE_SIZE

class PreProcessor():

    def __init__(self, train_csv: str, test_csv: str):
        self.df_train = pd.read_csv(train_csv)
        self.df_test = pd.read_csv(test_csv)
    
    def ellipsoid_fit(self, point_data: np.ndarray, mode: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = point_data[:, 0]
        Y = point_data[:, 1]
        Z = point_data[:, 2]

        # AlGEBRAIC EQUATION FOR ELLIPSOID, from CARTESIAN DATA - Form 1 
        if mode == 0:  # 9-DOF MODE
            D = np.array([X * X + Y * Y - 2 * Z * Z,
                        X * X + Z * Z - 2 * Y * Y,
                        2 * X * Y, 2 * X * Z, 2 * Y * Z,
                        2 * X, 2 * Y, 2 * Z,
                        1 + 0 * X]).T

        elif mode == 1:  # 6-DOF MODE (no rotation) - Form 2 
            D = np.array([X * X + Y * Y - 2 * Z * Z,
                        X * X + Z * Z - 2 * Y * Y,
                        2 * X, 2 * Y, 2 * Z,
                        1 + 0 * X]).T

        # THE RIGHT-HAND-SIDE OF THE LLSQ PROBLEM
        d2 = np.array([X * X + Y * Y + Z * Z]).T

        # SOLUTION TO NORMAL SYSTEM OF EQUATIONS
        u = np.linalg.solve(D.T.dot(D), D.T.dot(d2))

        # CONVERT BACK TO ALGEBRAIC FORM
        if mode == 0:  # 9-DOF-MODE
            a = np.array([u[0] + 1 * u[1] - 1])
            b = np.array([u[0] - 2 * u[1] - 1])
            c = np.array([u[1] - 2 * u[0] - 1])
            v = np.concatenate([a, b, c, u[2:, :]], axis=0).flatten()

        elif mode == 1:  # 6-DOF-MODE
            a = u[0] + 1 * u[1] - 1
            b = u[0] - 2 * u[1] - 1
            c = u[1] - 2 * u[0] - 1
            zs = np.array([0, 0, 0])
            v = np.hstack((a, b, c, zs, u[2:, :].flatten()))

        # PUT IN ALGEBRAIC FORM FOR ELLIPSOID
        A = np.array([[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], v[9]]])

        # FIND CENTRE OF ELLIPSOID
        centre = np.linalg.solve(-A[0:3, 0:3], v[6:9])

        # FORM THE CORRESPONDING TRANSLATION MATRIX
        T = np.eye(4)
        T[3, 0:3] = centre

        # TRANSLATE TO THE CENTRE, ROTATE
        R = T.dot(A).dot(T.T)

        # SOLVE THE EIGENPROBLEM
        evals, evecs = np.linalg.eig(R[0:3, 0:3] / -R[3, 3])

        # CALCULATE SCALE FACTORS AND SIGNS
        radii = np.sqrt(1 / abs(evals))
        sgns = np.sign(evals)
        radii *= sgns

        return (centre, evecs, radii)

    def reshape_data(self, is_train = True) -> Tuple[np.ndarray, np.ndarray]:

        df = self.df_train if is_train == True else self.df_test

        df = df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
            "iphoneMagX", "iphoneMagY", "iphoneMagZ",
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "orientW", "orientX", "orientY", "orientZ"
        ]]

        X = df[[
                "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
                "iphoneMagX", "iphoneMagY", "iphoneMagZ",
                "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            ]].to_numpy()

        y = df[[
            "orientW", "orientX", "orientY", "orientZ"
        ]].to_numpy()


        X_reshaped = np.zeros((REORIENT_NET_BATCH_SIZE, REORIENT_NET_LOOKBACK, 9))
        y_reshaped = np.zeros((REORIENT_NET_BATCH_SIZE, 4))

        for i in range(REORIENT_NET_BATCH_SIZE):
            y_position = i + REORIENT_NET_LOOKBACK
            X_reshaped[i] = X[i:y_position]
            y_reshaped[i] = y[y_position]

        return (X_reshaped, y_reshaped)

        



