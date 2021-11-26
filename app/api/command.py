import argparse
import sys
import os

from pydantic import ValidationError

from app.resources.constants import (
    COMMAND_LINE_OPTIONS, 
    COMMAND_LINE_DESCRIPTION, 
    CMD_PREPROCESS_DESCRIPTION, 
    CMD_TRAIN_DESCRIPTION,
    CMD_TEST_DESCRIPTION
)
from app.services.trainer import Trainer
from app.services.preprocessor import PreProcessor
from app.nn_models.nn_orient import ReOrientNet
from app.models.options import Option
from app.models.cmd_args import CommonArgs 
from app.utils.initializers import get_files, create_data, create_csv, create_test_data

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

    def preprocess(self):
        """Preprocess will create a csv file for training purposes
        """
        
        print("Running training set initializers...\n")

        parser = argparse.ArgumentParser(
            description = CMD_PREPROCESS_DESCRIPTION,
        )

        vars(parser.parse_args(sys.argv[2:]))
        print("Creating train csv...")

        train_file_path = "datasets/csvs/train.csv"

        if not os.path.exists(train_file_path):
            option = Option.TRAIN
            file_path = "datasets/csvs/train.csv"

            files = get_files(option)
            df = create_data(files=files)
            create_csv(df, file_path)


    def train(self):
        """Train is used to perform training with one of 
        the neural networks. There are three options to use here: 
        1 = Building 1
        2 = Building 2 
        3 = Building 3
        """

        parser = argparse.ArgumentParser(
            description = CMD_TRAIN_DESCRIPTION
        )
        parser.add_argument("--option", required=True)
        args = vars(parser.parse_args(sys.argv[2:]))

        try:
            train_args = CommonArgs.parse_obj(args)
        except ValueError as error:
            print(str(error))
            exit(1)


    def test(self):
        """Test is used to perform testing with one of the neural networks
        for that was trained for each building. There are three options to use here: 
        1 = Building 1
        2 = Building 2
        3 = Building 3
        """

        parser = argparse.ArgumentParser(
            description = CMD_TRAIN_DESCRIPTION
        )
        parser.add_argument("--option", required=True)
        args = vars(parser.parse_args(sys.argv[2:]))

        try:
            test_args = CommonArgs.parse_obj(args)
        except ValueError as error:
            print(str(error))
            exit(1)

