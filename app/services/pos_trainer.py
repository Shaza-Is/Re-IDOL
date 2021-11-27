import tensorflow as tf
import numpy as np
import pandas as pd
import random

from tensorflow.keras import Input, Model, layers
from typing import Generator


class PosTrainer(object):

    def __init__(self, model):
        self.model = model

    def generate_training_samples(self, matrix: np.ndarray) -> Generator[np.ndarray, None, None]:
        pass

    def compile_model(self) -> None:
        pass

    
    def train_model(self) -> None:
        
        seed = 10
        random.seed(seed)
        np.random.seed(seed)


    def display_model(self) -> str:

        return self.model.summary()

