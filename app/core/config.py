import os

REORIENT_NET_EPOCHS = os.getenv("REORIENT_NET_EPOCHS", 100)
REORIENT_NET_LEARNING_RATE = os.getenv("REORIENT_NET_LEARNING_RATE", 0.001)
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