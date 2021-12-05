from app.services.preprocessor import PreProcessor


COMMAND_LINE_DESCRIPTION = ":reIDOL CLI tool"
COMMAND_LINE_OPTIONS = """nn <command> [<args>]

The most commonly used nn commands are:
    train --option=(number 1-3)     Trains selected network, options range from 1 to 3. 
    test  --option=(number 1-3)     Tests selected network, options range from 1 to 3.

"""

# CMD_PREPROCESS_DESCRIPTION = '''This command will generate train csv file used specifically to train any of the neural networks'''
CMD_TRAIN_DESCRIPTION = "This command trains the neural network, it receives options 1, 2 or 3 to specify the building"
CMD_TEST_DESCRIPTION = "This command will test the neural network, it receives options 1, 2 or 3 to specify the building"

COMMON_ARGS_VALIDATION_ERROR_INCORRECT_NUMBER = "Error: option value must be 1, 2 or 3."
COMMON_ARGS_VALIDATION_ERROR_NOT_INT = "Error: option value must be int, cannot be float or double"
COMMON_ARGS_VALIDATION_ERROR_NOT_A_NUMBER = "Error: option value must be a number from 1 to 3"

REORIENT_INPUT_SIZE_COLS = 9
REORIENT_INPUT_SIZE_ROWS = 100
REORIENT_OUTPUT_LAYER_SIZE = 4

REORIENT_OPTIMIZER = "Adam"
# REORIENT_METRICS = "mape"

POS_NET_OPTIMIZER = "Adam"
POS_NET_LOSS = "mse"
# POS_NET_METRICS = "mape"
