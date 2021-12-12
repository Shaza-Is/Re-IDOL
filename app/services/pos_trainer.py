import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np
import pandas as pd
import random
import datetime

from tensorflow.keras import Input
from typing import Generator, Tuple, Dict
from pandas import DataFrame
from string import Template
from loguru import logger

from app.resources.constants import (
    POS_NET_LOSS
)
from app.core.config import (
    POS_NET_EPOCHS, 
    POS_NET_LEARNING_RATE,
    SEED
)
from app.nn_models.nn_position import build_position

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class PosTrainer(object):

    def __init__(self, building_num: int, df: DataFrame, is_reduced: bool = False):
        """Constructor for PosTrainer

        Args:
            building_num (int): the building number
            df (DataFrame): a data fram containing the data points. 
            is_reduced (bool, optional): used to do training on a reduced sample. Defaults to False.
        """
        self.building_num = building_num

        if is_reduced:
            length = int(df.shape[0] / 12)
            self.df = df[:length]
        else:
            self.df = df

        self._preprocess_data(self.df)



    def _preprocess_data(self, df: DataFrame):
        """_preprocess_data will take data frame and apply quaternion 
        rotations to the accelerometer and gyroscope data. 

        Args:
            df (DataFrame): a data frame containing the orientation and position data.
        """
        acc = tf.convert_to_tensor(df[["iphoneAccX", "iphoneAccY", "iphoneAccZ"]])
        gyro = tf.convert_to_tensor(df[["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"]])
        quats = tf.convert_to_tensor(df[["orientX", "orientY", "orientZ", "orientW"]])
        #print(acc[:5,:])

        new_acc = tfg.quaternion.rotate(acc, quats).numpy()
        new_gyro = tfg.quaternion.rotate(gyro, quats).numpy()
        #print(new_acc[:5,:])

        new_df_acc = pd.DataFrame(data=new_acc, columns=["iphoneAccX", "iphoneAccY", "iphoneAccZ"])
        new_df_gyro = pd.DataFrame(data=new_gyro, columns=["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"])

        df.loc[:,"iphoneAccX"] = new_df_acc["iphoneAccX"]
        df.loc[:,"iphoneAccY"] = new_df_acc["iphoneAccY"]
        df.loc[:,"iphoneAccZ"] = new_df_acc["iphoneAccZ"]

        df.loc[:,"iphoneGyroX"] = new_df_gyro["iphoneGyroX"]
        df.loc[:,"iphoneGyroY"] = new_df_gyro["iphoneGyroY"]
        df.loc[:,"iphoneGyroZ"] = new_df_gyro["iphoneGyroZ"]


    def _generate_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        """generate_samples creates a generator for the training samples for posnet

        Args:
            matrix (np.ndarray): [description]
            batch_size (int, optional): [description]. Defaults to 64.

        Yields:
            Generator[np.ndarray, None, None]: a generator that returns 100 x 6 matrices in batches of 64. 
        """

        while True:
            acc = np.array([row[0:3].tolist() for row in matrix])
            gyro = np.array([row[3:6].tolist() for row in matrix])
            pos = np.array([row[6:].tolist() for row in matrix])

            xa_batch = np.zeros((batch_size,100,3))
            xg_batch = np.zeros((batch_size,100,3))
            y_batch = np.zeros((batch_size,2))

            current_batch_number = 0
            for index in range(len(pos)):
                xa_batch[current_batch_number,:,:] = acc[index,:]
                xg_batch[current_batch_number,:,:] = gyro[index,:]
                y_batch[current_batch_number,:] = pos[index, :]

                current_batch_number += 1

                if current_batch_number >= batch_size:
                    current_batch_number = 0
                    yield([xa_batch, xg_batch], y_batch)

    def compile_model(self, latest_checkpoint: Tuple[str, int, float, float]) -> None:
        """compile_model builds a PosNet model, if latest checkpoint is available it will build a
        model from the last time the model was trained. 

        Args:
            latest_checkpoint (Tuple[str, int, float, float]): a tuple containing a file path to the checkpoint, 
            the last epoch, loss and metric. 
        """

        if latest_checkpoint:
            self.model = tf.keras.models.load_model(latest_checkpoint[0], compile = False)
        else:
            input1 = Input(shape=(100, 3), dtype="float32")
            input2 = Input(shape=(100, 3), dtype="float32")
            inputs = [input1, input2]

            self.model = build_position(inputs)

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = POS_NET_LEARNING_RATE),
            loss = POS_NET_LOSS,
            metrics = ["mse"]
        )

    
    def train_model(self, initial_epoch=0) -> None:
        """train_model generates the training sample and then 
        trains the PosNet model in batches of 64.

        Args:
            initial_epoch (int, optional): epoch number to start model training. defaults to 0.
        """

        save_file_path = f"saves/pos/building{self.building_num}"

        matrix = self.df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ",
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "processedPosX", "processedPosY"
        ]].to_numpy()
        generator = self._generate_samples(matrix)
        
        steps = len(self.df[["processedPosX", "processedPosY"]].to_numpy())
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = "logs/fit/pos{timestamp}".format(timestamp=timestamp)
        checkpoint = Template("saves/pos/checkpoints_pos_building${building_num}/${timestamp}_pos_chkpt_epoch_{epoch:03d}_loss_{loss:.4f}_metric_{mse:.4f}.hdf5")
        checkpoint = checkpoint.substitute(building_num=self.building_num, timestamp=timestamp)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint)

        self.model.fit(generator, 
            epochs=POS_NET_EPOCHS,
            initial_epoch=initial_epoch, 
            verbose=1, 
            steps_per_epoch=steps,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )

        self.model.save(save_file_path)

    def display_model(self) -> None:
        """display_model returns PosNet's summary that 
        specifies the layers in the model along with other 
        information such as parameters. 
        """
        self.model.summary()

    def evaluate_model(self, trajectories: Dict[int, DataFrame]) -> None:
        """evaluate_model tests the model by running 

        Args:
            trajectories (Dict[int, DataFrame]): [dictionary with trajectory dataframes]
        """
        self.model = tf.keras.models.load_model(f"saves/pos/building{self.building_num}", compile=False)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = POS_NET_LEARNING_RATE),
            loss = POS_NET_LOSS)
        self.display_model()

        for trajectory_num, trajectory in trajectories.items():
            self._preprocess_data(trajectory)

            matrix = trajectory[[
                "iphoneAccX", "iphoneAccY", "iphoneAccZ",
                "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
                "processedPosX", "processedPosY"
            ]].to_numpy()

            steps = len(trajectory[["processedPosX", "processedPosY"]].to_numpy())
            generator = self._generate_samples(matrix)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"logs/evaluate/pos_trajectory_{trajectory_num}_{timestamp}".format(timestamp=timestamp, trajectory_num=trajectory_num)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            logger.info(f"PosNet testing on trajectory {trajectory_num}")

            self.model.evaluate(generator, steps=steps, verbose=1, callbacks=[tensorboard_callback])
