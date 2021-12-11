import os

SEED = os.getenv("SEED", 13)
REORIENT_NET_EPOCHS = os.getenv("REORIENT_NET_EPOCHS", 20)
REORIENT_NET_LEARNING_RATE = os.getenv("REORIENT_NET_LEARNING_RATE", 0.0005)
REORIENT_NET_BATCH_SIZE = os.getenv("REORIENT_NET_BATCH_SIZE", 64)
REORIENT_NET_LOOKBACK = os.getenv("REORIENT_NET_LOOKBACK", 20)

REORIENT_NET_LSTM_1 = os.getenv("REORIENT_NET_LSTM_1", 100)
REORIENT_NET_LSTM_2 = os.getenv("REORIENT_NET_LSTM_2", 100)
REORIENT_NET_DENSE_1 = os.getenv("REORIENT_NET_DENSE_1", 100)
REORIENT_NET_DENSE_2 = os.getenv("REORIENT_NET_DENSE_2", 32)
REORIENT_NET_DENSE_3 = os.getenv("REORIENT_NET_DENSE_3", 100)
REORIENT_NET_DENSE_4 = os.getenv("REORIENT_NET_DENSE_4", 32)
REORIENT_NET_OUTPUT_1 = os.getenv("REORIENT_OUTPUT_1", 4)
REORIENT_NET_OUTPUT_2 = os.getenv("REORIENT_OUTPUT_1", 6)

REORIENT_DENSE_ACTIVATION = os.getenv("REORIENT_DENSE_ACTIVATION", "tanh")

POS_NET_EPOCHS = os.getenv("POS_NET_EPOCHS", 20)
POS_NET_LEARNING_RATE = os.getenv("POS_NET_LEARNING_RATE", 0.001)
POS_NET_BATCH_SIZE = os.getenv("POS_NET_BATCH_SIZE", 64)

POS_NET_LSTM_1 = os.getenv("POS_NET_LSTM_1", 100)
POS_NET_LSTM_2 = os.getenv("POS_NET_LSTM_2", 100)
POS_NET_DENSE_1 = os.getenv("POS_NET_DENSE_1", 100)
POS_NET_DENSE_2 = os.getenv("POS_NET_DENSE_2", 20)
POS_NET_OUTPUT_1 = os.getenv("POS_NET_OUTPUT_1", 2)

POS_NET_ACTIVATION_1 = os.getenv("POS_NET_ACTIVATION_1", "tanh")
POS_NET_ACTIVATION_2 = os.getenv("POS_NET_ACTIVATION_2", "linear")