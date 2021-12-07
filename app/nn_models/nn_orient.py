from tensorflow.keras import Model, Input, layers
from typing import List


from app.core.config import (
    REORIENT_NET_LSTM_1, 
    REORIENT_NET_LSTM_2, 
    REORIENT_NET_DENSE_1,
    REORIENT_NET_DENSE_2,
    REORIENT_NET_DENSE_3,
    REORIENT_NET_DENSE_4,
    REORIENT_NET_OUTPUT_1,
    REORIENT_NET_OUTPUT_2,
    REORIENT_DENSE_ACTIVATION
)

def build_reorient(inputs: List[Input]) -> Model:
        input_layer = layers.Concatenate()([inputs[0], inputs[1], inputs[2]])

        lstm_layer_1 = layers.LSTM(REORIENT_NET_LSTM_1,  kernel_initializer = "glorot_uniform", 
            recurrent_initializer = "orthogonal", return_sequences=True)(input_layer)
        lstm_layer_2 = layers.LSTM(REORIENT_NET_LSTM_2, kernel_initializer = "glorot_uniform", 
            recurrent_initializer = "orthogonal", return_sequences=False)(lstm_layer_1)

        dense_layer_1 = layers.Dense(units=REORIENT_NET_DENSE_1,
            activation=REORIENT_DENSE_ACTIVATION)(lstm_layer_2)
        dense_layer_2 = layers.Dense(units=REORIENT_NET_DENSE_2, 
            activation=REORIENT_DENSE_ACTIVATION)(dense_layer_1)

        dense_layer_3 = layers.Dense(units=REORIENT_NET_DENSE_3,
            activation=REORIENT_DENSE_ACTIVATION)(lstm_layer_2)
        dense_layer_4 = layers.Dense(units=REORIENT_NET_DENSE_4,
            activation=REORIENT_DENSE_ACTIVATION)(dense_layer_3)

        theta_output = layers.Dense(units=REORIENT_NET_OUTPUT_1)(dense_layer_2)
        sigma_output = layers.Dense(units=REORIENT_NET_OUTPUT_2)(dense_layer_4)
        
        output = layers.Concatenate()([theta_output, sigma_output])

        return Model([inputs[0], inputs[1], inputs[2]], [output])
