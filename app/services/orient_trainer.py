import tensorflow as tf
import numpy as np
import pandas as pd
import random

from tensorflow.keras import Input, Model, layers
from typing import Generator

from app.resources.constants import (
    REORIENT_OPTIMIZER, 
    REORIENT_METRICS
)
from app.core.config import REORIENT_NET_EPOCHS
from app.nn_models.nn_orient_loss import ReOrientLoss

from pprint import pprint

class OrientTrainer(object):

    def __init__(self, model: Model):
        self.model = model

    def _generate_training_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        """generate_training_samples takes a matrix and separates into 
        batches to provide to the neural network. These batches are separated 
        into sizes of 64 x 100. The values are divided as follows: 

        Independent Variables:
        - Magnetometer Readings
        - Gyroscope Readings
        - Accelerometer  Readings 

        Dependent Variables: 
        - Orientation Quaternion - (W, X, Y, Z) 

        Args:
            matrix (np.ndarray): Data taken from training information. 
            batch_size (int, optional): Size of batches. Defaults to 64.
        """
        while True:
            acc = np.array([row[0:3].tolist() for row in matrix])
            gyro = np.array([row[3:6].tolist() for row in matrix])
            mag = np.array([row[6:9].tolist() for row in matrix])
            orient = np.array([row[9:].tolist() for row in matrix])

            xa_batch = np.zeros((batch_size,100,3))
            xg_batch = np.zeros((batch_size,100,3))
            xm_batch = np.zeros((batch_size,100,3))
            y_theta_batch = np.zeros((batch_size,4))
            y_sigma_batch = np.finfo(np.float32).eps* np.ones((batch_size,6)) ## To remove
            y_batch = np.concatenate((y_theta_batch, y_sigma_batch), axis=1)
            
            current_batch_number = 0
            for index in range(len(orient)):

                xa_batch[current_batch_number,:,:] = acc[index,:]
                xg_batch[current_batch_number,:,:] = gyro[index,:]
                xm_batch[current_batch_number,:,:] = mag[index,:]
                y_theta_batch[current_batch_number,:4] = orient[index,:]
                current_batch_number += 1
                if current_batch_number >= batch_size:
                    current_batch_number = 0              
                    yield([xa_batch, xg_batch, xm_batch],[y_batch])
    
    def compile_model(self) -> None:
        """compile_model compiles the tensorflow model
        """
        self.model.compile(
            optimizer = REORIENT_OPTIMIZER,
            loss = ReOrientLoss(),
            metrics = [REORIENT_METRICS]
        )

    def train_model(self) -> None:
        """train_model generates the training samples and then
        trains the models in batches
        """
        seed = 10
        random.seed(seed)
        np.random.seed(seed)

        df = pd.read_csv("datasets/csvs/train.csv")
        matrix = df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "iphoneMagX", "iphoneMagY", "iphoneMagZ",
            "orientX", "orientY", "orientZ", "orientW"
        ]].to_numpy()

        steps = len(df[["orientX", "orientY", "orientZ", "orientW"]].to_numpy())
        
        generator = self._generate_training_samples(matrix)
        
        self.model.fit(generator, epochs=REORIENT_NET_EPOCHS, verbose=2, steps_per_epoch=steps)

    def display_model(self) -> str:
        """display_model will return the model's summary. 

        Returns:
            [str]: a string with the model summary
        """
        return self.model.summary()

