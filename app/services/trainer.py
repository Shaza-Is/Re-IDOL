import tensorflow as tf

from tensorflow.keras import Input, Model, layers

from app.resources.constants import (
    REORIENT_OPTIMIZER, 
    REORIENT_LOSS, 
    REORIENT_METRICS
)
from app.core.config import REORIENT_NET_EPOCHS

class Trainer(object):

    def __init__(self, model: Model):
        self.model = model

    
    def compile_model(self):
        self.model.compile(
            optimizer = REORIENT_OPTIMIZER,
            loss = REORIENT_LOSS,
            metrics= [REORIENT_METRICS]
        )

    def train_model(self, X, y):
        self.model.fit(X, y, epochs = REORIENT_NET_EPOCHS, verbose=2)

    def display_model(self):
        print(self.model.summary())

