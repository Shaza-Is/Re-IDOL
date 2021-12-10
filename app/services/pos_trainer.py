import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np
import pandas as pd
import random
import datetime

from tensorflow.keras import Input
from typing import Generator, Tuple
from pandas import DataFrame
from string import Template

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
        self.building_num = building_num

        if is_reduced:
            length = int(df.shape[0] / 128)
            self.df = df[:length]
        else:
            self.df = df

        acc = tf.convert_to_tensor(self.df[["iphoneAccX", "iphoneAccY", "iphoneAccZ"]].to_numpy())
        gyro = tf.convert_to_tensor(self.df[["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"]].to_numpy())
        quats = tf.convert_to_tensor(self.df[["orientX", "orientY", "orientZ", "orientW"]].to_numpy())

        new_acc = tfg.quaternion.rotate(acc, quats).numpy()
        new_gyro = tfg.quaternion.rotate(gyro, quats).numpy()

        new_df_acc = pd.DataFrame(data=new_acc, columns=["iphoneAccX", "iphoneAccY", "iphoneAccZ"])
        new_df_gyro = pd.DataFrame(data=new_gyro, columns=["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"])

        self.df.loc[:,"iphoneAccX"] = new_df_acc["iphoneAccX"]
        self.df.loc[:,"iphoneAccY"] = new_df_acc["iphoneAccY"]
        self.df.loc[:,"iphoneAccZ"] = new_df_acc["iphoneAccZ"]

        self.df.loc[:,"iphoneGyroX"] = new_df_gyro["iphoneGyroX"]
        self.df.loc[:,"iphoneGyroY"] = new_df_gyro["iphoneGyroY"]
        self.df.loc[:,"iphoneGyroZ"] = new_df_gyro["iphoneGyroZ"]


    def _generate_samples(self, matrix: np.ndarray, batch_size: int = 64) -> Generator[np.ndarray, None, None]:
        """generate_samples creates a generator for the training samples for posnet

        Args:
            matrix (np.ndarray): [description]
            batch_size (int, optional): [description]. Defaults to 64.

        Yields:
            Generator[np.ndarray, None, None]: [description]
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
        """compile_model compiles the tensorflow model
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
            loss = POS_NET_LOSS
        )

    
    def train_model(self, initial_epoch=0) -> None:
        """train_model generates the training sample and then 
        trains the model in batches of 64

        Args:
            initial_epoch (int, optional): [Epoch number to start model training]. Defaults to 0.
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

    def display_model(self) -> str:
        return self.model.summary()

    def evaluate_model(self) -> None:
        pass
