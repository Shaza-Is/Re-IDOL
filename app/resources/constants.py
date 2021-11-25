from app.services.preprocessor import PreProcessor


COMMAND_LINE_DESCRIPTION = ":reIDOL CLI tool"
COMMAND_LINE_OPTIONS = """nn <command> [<args>]

The most commonly used nn commands are: 
    train       Trains network by using .env file parameters.
    predict     Receives data and returns a prediction.

"""

CMD_TRAIN_DESCRIPTION = "This command trains the neural network"
CMD_PREDICT_DESCRIPTION = ""

REORIENT_FEATURE_SIZE = 9
REORIENT_OUTPUT_LAYER_SIZE = 4

REORIENT_OPTIMIZER = "Adam"
REORIENT_LOSS = "sparse_categorical_crossentropy"
REORIENT_METRICS = "accuracy"