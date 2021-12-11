import tensorflow as tf
import numpy as np
import random
import os
import datetime

from loguru import logger
from tensorflow.keras import Input
from typing import Generator, Tuple, Dict
from pandas import DataFrame
from string import Template

from app.core.config import (
    REORIENT_NET_EPOCHS, 
    REORIENT_NET_LEARNING_RATE,
    SEED
)
from app.nn_models.nn_orient_loss import ReOrientLoss, quat_metric, MyLossRMSE
from app.nn_models.nn_orient import build_reorient
        
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class OrientTrainer(object):

    def __init__(self, building_num: int, df: DataFrame, is_reduced: bool = False):
        """constructor for OrientTrainer

        Args:
            building_num (int): the building number.
            df (DataFrame): the data to train/test the network.
            is_reduced (bool, optional): used to do training on a reduced sample. Defaults to False.
        """
        self.building_num = building_num

        if is_reduced: 
            length = int(df.shape[0] / 128)
            self.df = df[:length]
        else: 
            self.df = df


    def _generate_training_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        """_generate_training_samples creates a generator with batches to provide data
        to the training function

        Args:
            matrix (np.ndarray): [matrix with data to input to OrientNet]
            batch_size (int, optional): [batch size of 64]. Defaults to 64.

        Yields:
            Generator[np.ndarray, None, None]: a generator that returns 100 x 9 matrices in batches of 64
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
            
            current_batch_number = 0
            for index in range(len(orient)):

                xa_batch[current_batch_number,:,:] = acc[index,:]
                xg_batch[current_batch_number,:,:] = gyro[index,:]
                xm_batch[current_batch_number,:,:] = mag[index,:]
                y_theta_batch[current_batch_number,:4] = orient[index,:]
                

                current_batch_number += 1

                if current_batch_number >= batch_size:
                    current_batch_number = 0             
                    yield([xa_batch, xg_batch, xm_batch],[y_theta_batch])
        
    
    def _generate_prediction_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        """_generate_prediction_samples creates a generator with batches to provide data
        to the prediction function

        Args:
            matrix (np.ndarray): [matrix with data to input to OrientNet]
            batch_size (int, optional): [batch size of 64]. Defaults to 64.

        Yields:
            Generator[np.ndarray, None, None]: [a generator that returns 100 x 9 matrices in batches of 64]
        """        
        while True:
            acc = np.array([row[0:3].tolist() for row in matrix])
            gyro = np.array([row[3:6].tolist() for row in matrix])
            mag = np.array([row[6:].tolist() for row in matrix])

            xa_batch = np.zeros((batch_size,100,3))
            xg_batch = np.zeros((batch_size,100,3))
            xm_batch = np.zeros((batch_size,100,3))
            
            current_batch_number = 0
            for index in range(matrix.shape[0]):

                xa_batch[current_batch_number,:,:] = acc[index,:]
                xg_batch[current_batch_number,:,:] = gyro[index,:]
                xm_batch[current_batch_number,:,:] = mag[index,:]
                

                current_batch_number += 1

                if current_batch_number >= batch_size:
                    current_batch_number = 0             
                    yield [[xa_batch, xg_batch, xm_batch]]

    def compile_model(self, latest_checkpoint: Tuple[str, int, float, float]) -> None:
        """compile_model builds an OrientNet model, if latest checkpoint is available it will build a
        model from the last time the model was trained. 

        Args:
            latest_checkpoint (Tuple[str, int, float, float]): a tuple containing a file path to the checkpoint, \
            the last epoch, loss and metric. 
        """

        if latest_checkpoint:
            self.model = tf.keras.models.load_model(latest_checkpoint[0], compile = False)
        else:
            input1 = Input((100, 3), dtype="float32")
            input2 = Input((100, 3), dtype="float32")
            input3 = Input((100, 3), dtype="float32")

            inputs = [input1, input2, input3]
            self.model = build_reorient(inputs)

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = REORIENT_NET_LEARNING_RATE),
            metrics= [quat_metric],
            loss = ReOrientLoss()
        )

    def train_model(self, initial_epoch=0) -> None:
        """train_model generates the training samples and then
        trains the OrientNet model in batches of 64

        Args:
            initial_epoch (int, optional): epoch number to start model training. defaults to 0. 
        """

        matrix = self.df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "iphoneMagX", "iphoneMagY", "iphoneMagZ",
            "orientX", "orientY", "orientZ", "orientW"
        ]].to_numpy()

        steps = len(self.df[["orientX", "orientY", "orientZ", "orientW"]].to_numpy())
        
        generator = self._generate_training_samples(matrix)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = "logs/fit/orient{timestamp}".format(timestamp=timestamp)
        checkpoint = Template("saves/orient/checkpoints_orient_building${building_num}/${timestamp}_orient_chkpt_epoch_{epoch:03d}_loss_{loss:.4f}_metric_{metric_quat_diff:.4f}.hdf5")
        checkpoint = checkpoint.substitute(building_num=self.building_num, timestamp=timestamp)


        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint)


        self.model.fit(
            generator, 
            epochs=REORIENT_NET_EPOCHS,
            initial_epoch=initial_epoch, 
            verbose=1, 
            steps_per_epoch=steps,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )


    def evaluate_model(self, trajectories: Dict[int, DataFrame], latest_checkpoint: Tuple[str, int, float, float]) -> None:
        """evaluate_model evaluates OrientNet against each trajectory in the datasets. 

        Args:
            trajectories (Dict[int, DataFrame]): a dictionary that contains each individual trajectory
            latest_checkpoint (Tuple[str, int, float, float]): a checkpoint to load the last model that was trained
        """         
        self.model = tf.keras.models.load_model(latest_checkpoint[0], compile=False)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = REORIENT_NET_LEARNING_RATE),
            loss = MyLossRMSE())
        self.display_model()

        for trajectory_num, trajectory in trajectories.items():

            matrix = trajectory[[
                "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
                "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
                "iphoneMagX", "iphoneMagY", "iphoneMagZ",
                "orientX", "orientY", "orientZ", "orientW"
            ]].to_numpy()

            steps = len(trajectory[["orientX", "orientY", "orientZ", "orientW"]].to_numpy())
            generator = self._generate_training_samples(matrix)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = "logs/evaluate/orient_trajectory_{trajectory_num}_{timestamp}".format(timestamp=timestamp, trajectory_num=trajectory_num)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            logger.info(f"OrientNet testing on trajectory {trajectory_num}")

            self.model.evaluate(generator, steps=steps, verbose=1, callbacks=[tensorboard_callback])

    def predict(self) -> np.ndarray:
        """predict returns a number of predictions based on 
        data that is fed to it. 

        Returns:
            np.ndarray: output matrix with predictions made
        """

        matrix = self.df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "iphoneMagX", "iphoneMagY", "iphoneMagZ",
        ]].to_numpy()

        steps = matrix.shape[0]
        generator = self._generate_prediction_samples(matrix)

        return self.model.predict(generator, steps=steps, verbose=1)
   

    def display_model(self) -> str:
        """display_model returns OrientNet's summary that 
        specifies the layers in the model along with other 
        information such as parameters. 
        """
        self.model.summary()

