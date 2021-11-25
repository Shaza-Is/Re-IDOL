from tensorflow.keras import Model, Input, layers

from app.core.config import (
    REORIENT_NET_LSTM_1, 
    REORIENT_NET_LSTM_2, 
    REORIENT_NET_FC_1,
    REORIENT_NET_FC_2,
    REORIENT_NET_BATCH_SIZE
)
from app.resources.constants import REORIENT_FEATURE_SIZE, REORIENT_OUTPUT_LAYER_SIZE

class ReOrientNet(Model):
    def __init__(self):
        super(ReOrientNet, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=(REORIENT_FEATURE_SIZE,), batch_size=REORIENT_NET_BATCH_SIZE, name="ReOrientNet_Input")
        self.lstm_layer_1 = layers.LSTM(units=REORIENT_NET_LSTM_1, name="ReOrientNet_LSTM_1", return_sequences=True, return_state=True)
        self.lstm_layer_2 = layers.LSTM(units=REORIENT_NET_LSTM_2, name="ReOrientNet_LSTM_2", return_sequences=True, return_state=True)
        self.fc_layer_1 = layers.Dense(units=REORIENT_NET_FC_1, name="ReOrientNet_FC_1")
        self.fc_layer_2 = layers.Dense(units=REORIENT_NET_FC_2, name="ReOrientNet_FC_2")
        self.fc_layer_3 = layers.Dense(units=100, name="ReOrientNet_FC_3")
        self.fc_layer_4 = layers.Dense(units=100, name="ReOrientNet_FC_4")
        self.output_layer = layers.Dense(units=5, name="ReOrientNet_Output")


    def call(self, inputs: Input, is_training: bool = False):
        import pdb
        x = self.lstm_layer_1(x)
        x = self.lstm_layer_2(x)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        return self.output_layer(x)