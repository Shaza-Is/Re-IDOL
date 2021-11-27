import tensorflow as tf
import numpy as np
import pandas as pd
import random

from tensorflow.keras import Input, Model, layers
from typing import Generator


from app.resources.constants import (
    POS_NET_OPTIMIZER,
    POS_NET_LOSS,
    POS_NET_METRICS,
)
from app.core.config import POS_NET_EPOCHS

class PosTrainer(object):

    def __init__(self, model):
        self.model = model

    def _generate_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        while True:
            acc = np.array([row[0:3].tolist() for row in matrix])
            gyro = np.array([row[3:6].tolis() for row in matrix])
            pos = np.array([row[6:].tolist() for row in matrix])

            xa_batch = np.zeros((batch_size,100,3))
            xg_batch = np.zeros((batch_size,100,3))
            y_batch = np.zeros((batch_size,2))

            current_batch_number = 0
            for index in range(len(pos)):
                xa_batch[current_batch_number,:,:] = acc[index,:]
                xg_batch[current_batch_number,:,:] = gyro[index,:]
                y_batch[current_batch_number,:,:] = pos[index, :]

                current_batch_number += 1

                if current_batch_number >= batch_size:
                    current_batch_number = 0
                    yield([xa_batch, xg_batch], y_batch)

    def compile_model(self) -> None:
        """compile_model compiles the tensorflow model
        """
        self.model.compile(
            optimizer = POSNET_OPTIMIZER,
            loss = POSNET_LOSS,
            metrics = [POSNET_METRICS]
        )

    
    def train_model(self) -> None:
        
        seed = 10
        random.seed(seed)
        np.random.seed(seed)

        df = pd.read_csv("datasets/csvs/train.csv")
        pos_data = df[[
            "processedPosX", "processedPosY"
        ]].to_numpy()

        matrix = np.hstack([np.random.rand(matrix.shape[0], 6), pos_data])
        generator = self._generate_samples(matrix)
        
        steps = len(pos_data)

        self.model.fit(generator, epochs=POSNET, verbos=2, steps_per_epoch=steps)

    def display_model(self) -> str:
        return self.model.summary()

    def evaluate_model(self) -> None:

