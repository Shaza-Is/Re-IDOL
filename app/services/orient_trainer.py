import tensorflow as tf
import numpy as np
import random
import os
import datetime

from tensorflow.keras import Input
from typing import Generator
from pandas import DataFrame

from app.core.config import (
    REORIENT_NET_EPOCHS, 
    REORIENT_NET_LEARNING_RATE
)
from app.nn_models.nn_orient_loss import ReOrientLoss
import app.nn_models.nn_orient_loss
from app.nn_models.nn_orient import build_reorient



class OrientTrainer(object):

    def __init__(self, building_num: int, df: DataFrame, is_reduced: bool = False):
        self.building_num = int(building_num)

        if is_reduced: 
            length = int(df.shape[0] / 128)
            self.df = df[:length]
        else: 
            self.df = df


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
        
    
    def compile_model(self, latest_checkpoint: str = "") -> None:
        """compile_model compiles the tensorflow model
        """

        if latest_checkpoint:
            self.model = tf.keras.models.load_model(latest_checkpoint, compile = False)
        else:
            input1 = Input((100, 3), dtype="float32")
            input2 = Input((100, 3), dtype="float32")
            input3 = Input((100, 3), dtype="float32")

            inputs = [input1, input2, input3]
            self.model = self.model = build_reorient(inputs)

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = REORIENT_NET_LEARNING_RATE),
            metrics= [app.nn_models.nn_orient_loss.quat_metric],
            loss = ReOrientLoss()
        )

    def train_model(self) -> None:
        """train_model generates the training samples and then
        trains the models in batches
        """
        seed=100
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
        save_file_path = f"saves/orient/building{self.building_num}"

        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

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
        checkpoint = """saves/orient/checkpoints_orient_building{building_num}/
            orient{timestamp}.hdf5""".format(building_num=self.building_num, timestamp=timestamp)


        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint)


        self.model.fit(
            generator, 
            epochs=REORIENT_NET_EPOCHS, 
            verbose=1, 
            steps_per_epoch=steps,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )

        self.model.save(save_file_path)

    def evaluate_model(self) -> None: 
        self.model = tf.keras.models.load_model("saves/orient/building1", compile=False)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = REORIENT_NET_LEARNING_RATE),
            loss = ReOrientLoss())
        self.display_model()

        matrix = self.df[[
            "iphoneAccX", "iphoneAccY", "iphoneAccZ", 
            "iphoneGyroX", "iphoneGyroY", "iphoneGyroZ",
            "iphoneMagX", "iphoneMagY", "iphoneMagZ",
            "orientX", "orientY", "orientZ", "orientW"
        ]].to_numpy()

        steps = len(self.df[["orientX", "orientY", "orientZ", "orientW"]].to_numpy())
        generator = self._generate_training_samples(matrix)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/evaluate/orient{timestamp}".format(timestamp=timestamp)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.evaluate(generator, steps=steps, verbose=1, callbacks=[tensorboard_callback])

    def display_model(self) -> str:
        """display_model will return the model's summary. 

        Returns:
            [str]: a string with the model summary
        """
        return self.model.summary()

