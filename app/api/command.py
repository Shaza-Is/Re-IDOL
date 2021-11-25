import argparse
import sys
import os

from app.resources.constants import (
    COMMAND_LINE_OPTIONS, 
    COMMAND_LINE_DESCRIPTION, 
    CMD_TRAIN_DESCRIPTION
)

from app.services.trainer import Trainer
from app.services.preprocessor import PreProcessor
from app.nn_models.nn_orient import ReOrientNet
from app.utils.initializers import get_files, create_csv, create_data

class CommandLine(object):
    """This is the command line interface 
    for the neural network. This is how the functionality 
    of the NN will be accessed.

    Args:
        object (object): default python object
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description = COMMAND_LINE_DESCRIPTION ,
            usage = COMMAND_LINE_OPTIONS
        )

        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        
        getattr(self, args.command)()

    def train(self):
        """Train is the command to use in order to train 
        the neural network 
        """
        
        print("Running training set initializers...\n")

        parser = argparse.ArgumentParser(
            description = CMD_TRAIN_DESCRIPTION,
        )

        vars(parser.parse_args(sys.argv[2:]))
        
        print("Creating train and test csvs...")

        # TODO: Refactor lines 56-68

        train_file_path = "datasets/csvs/train.csv"
        test_file_path = "datasets/csvs/building1.csv"


        if not os.path.exists(train_file_path):
            train_files = get_files(option = 4)
            train_df = create_data(files = train_files)
            create_csv(train_df, "datasets/csvs/train.csv")
        
        if not os.path.exists(test_file_path):
            test_files = get_files(option = 1)
            test_df = create_data(files = test_files)
            create_csv(test_df, "datasets/csvs/building1.csv")

        pre = PreProcessor(train_csv=train_file_path, test_csv=test_file_path)
        model = ReOrientNet()
        trainer = Trainer(model)

        print("Preprocessing data....")

        (X_train, y_train) = pre.reshape_data(is_train = True)
        (X_test, y_test) = pre.reshape_data(is_train = False)

        print("Running nn training, create .env file to change hyper parameters.")

        trainer.compile_model()
        trainer.train_model(X_train, y_train)




    def predict(self):
        pass