from tensorflow.keras import Model, Input, layers
from typing import List

from app.core.config import (
    POS_NET_LSTM_1,
    POS_NET_LSTM_2,
    POS_NET_DENSE_1,
    POS_NET_DENSE_2,
    POS_NET_OUTPUT_1,
    POS_NET_ACTIVATION_1,
POS_NET_ACTIVATION_2
)

def create_position(inputs: List[Input]):

    input_layer = layers.Concatenate()([inputs[0], inputs[1]])

    lstm_layer_1 = layers.Bidirectional(layers.LSTM(POS_NET_LSTM_1, return_sequences=True))(input_layer)
    lstm_layer_2 = layers.Bidirectional(layers.LSTM(POS_NET_LSTM_2))(lstm_layer_1)
    dense_layer_1 = layers.Dense(units=POS_NET_DENSE_1, activation=POS_NET_ACTIVATION_1)(lstm_layer_2)
    dense_layer_2 = layers.Dense(units=POS_NET_DENSE_2, activation=POS_NET_ACTIVATION_1)(dense_layer_1)
    outputs = layers.Dense(units=POS_NET_OUTPUT_1, activation=POS_NET_ACTIVATION_2)(dense_layer_2)

    return Model([inputs[0], inputs[1], [outputs]])

