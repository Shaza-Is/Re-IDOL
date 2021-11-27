from tensorflow.keras import Model, Input, layers

from app.core.config import (
    REORIENT_NET_LSTM_1, 
    REORIENT_NET_LSTM_2, 
    REORIENT_NET_DENSE_1,
    REORIENT_NET_DENSE_2,
    REORIENT_NET_DENSE_3,
    REORIENT_NET_DENSE_4,
    REORIENT_NET_OUTPUT_1,
    REORIENT_NET_OUTPUT_2,
    REORIENT_NET_BATCH_SIZE,
    REORIENT_DENSE_ACTIVATION
)
from app.resources.constants import (
    REORIENT_INPUT_SIZE_COLS, 
    REORIENT_INPUT_SIZE_ROWS, 
    REORIENT_OUTPUT_LAYER_SIZE
)

class ReOrientNet(Model):
    def __init__(self):
        super(ReOrientNet, self).__init__()

        self.lstm_layer_1 = layers.LSTM(REORIENT_NET_LSTM_1, return_sequences=True, name="ReOrient_LSTM_1")
        self.lstm_layer_2 = layers.LSTM(REORIENT_NET_LSTM_2, return_sequences=False, name="ReOrient_LSTM_2")
        self.dense_layer_1 = layers.Dense(units=REORIENT_NET_DENSE_1, 
            activation=REORIENT_DENSE_ACTIVATION, name="ReOrient_Dense_1")
        self.dense_layer_2 = layers.Dense(units=REORIENT_NET_DENSE_2, 
            activation=REORIENT_DENSE_ACTIVATION, name="ReOrient_Dense_2")
        self.dense_layer_3 = layers.Dense(units=REORIENT_NET_DENSE_3,
            activation=REORIENT_DENSE_ACTIVATION,  name="ReOrient_Dense_3")
        self.dense_layer_4 = layers.Dense(units=REORIENT_NET_DENSE_4,
            activation=REORIENT_DENSE_ACTIVATION,  name="ReOrient_Dense_4")

        self.output_layer_1 = layers.Dense(units=REORIENT_NET_OUTPUT_1, name="ReOrient_Quaternion")
        self.output_layer_2 = layers.Dense(units=REORIENT_NET_OUTPUT_2, name="ReOrient_Cov_Matrix")

    def call(self, inputs: Input, is_training: bool = False):
        x = layers.Concatenate()([inputs[0], inputs[1], inputs[2]])

        x = self.lstm_layer_1(x)
        x = self.lstm_layer_2(x)

        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        theta = self.output_layer_1(x)
        
        x = self.dense_layer_3(x)
        x = self.dense_layer_4(x)
        sigma = self.output_layer_2(x)

        return layers.Concatenate()([theta, sigma])
        