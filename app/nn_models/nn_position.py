from tensorflow.keras import Model, Input, layers

from app.core.config import (
    POS_NET_LSTM_1,
    POS_NET_LSTM_2,
    POS_NET_DENSE_1,
    POS_NET_DENSE_2,
    POS_NET_OUTPUT_1,
    POS_NET_ACTIVATION_1,
    POS_NET_ACTIVATION_2
)

class PosNet(Model):
    def __init__(self):
        super(RePositionNet, self).__init__()
        
        self.lstm_layer_1 = layers.Bidirectional(layers.LSTM(POS_NET_LSTM_1, return_sequences=True))
        self.lstm_layer_2 = layers.Bidirectional(layers.LSTM(POS_NET_LSTM_2))
        self.dense_layer_1 = layers.Dense(units=POS_NET_DENSE_1, activation=POS_NET_ACTIVATION_1)
        self.dense_layer_2 = layers.Dense(units=POS_NET_DENSE_2, activation=POS_NET_ACTIVATION_1)
        self.output_layer_1 = layers.Dense(units=POS_NET_OUTPUT_1, activation=POS_NET_ACTIVATION_2)

    def call(self, inputs: Input, is_training: bool = False):
        x = layers.Concatenate([inputs[0], inputs[1]])

        x = self.lstm_layer_1(x)
        x = self.lstm_layer_2(x)
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        output = self.output_layer_1(x)

        return output

