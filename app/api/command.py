import argparse
import sys
import os

from loguru import logger

from app.resources.constants import (
    COMMAND_LINE_OPTIONS, 
    COMMAND_LINE_DESCRIPTION, 
    CMD_PREPROCESS_DESCRIPTION, 
    CMD_TRAIN_DESCRIPTION,
    CMD_TEST_DESCRIPTION
)
from app.services.orient_trainer import OrientTrainer
from app.services.pos_trainer import PosTrainer
from app.services.tensorboard_process import TensorboardSupervisor
from app.nn_models.nn_orient import ReOrientNet
from app.nn_models.nn_position import PosNet
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
        
        logger.info("Running training set initializers...")

        parser = argparse.ArgumentParser(
            description = CMD_PREPROCESS_DESCRIPTION,
        )

        vars(parser.parse_args(sys.argv[2:]))
        logger.info("Creating train csv...")

        train_file_path = "datasets/csvs/train.csv"

        if not os.path.exists(train_file_path):
            option = Option.TRAIN
            file_path = "datasets/csvs/train.csv"

            files = get_files(option)
            df = create_data(files=files)
            create_csv(df, file_path)

        logger.info("Train file created.")

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
        tensorboard_log_path = "logs/fit"

        try:
            train_args = CommonArgs.parse_obj(args)
            logger.info("Attempting to train ReOrient Net {option}".format(option=train_args.option))
            logger.info("Starting Tensorboard server at http://localhost:6006")
            
            tb_sup = TensorboardSupervisor(tensorboard_log_path)
            model = ReOrientNet()
            trainer = OrientTrainer(model)
            trainer.compile_model()
            trainer.train_model()

            logger.info("ReOrient Net training finished. Model weights have been saved.")
            logger.info("Attempting to train Pos Net {option}".format(option=train_args.option))
            
            model2 = PosNet()
            trainer2 = PosTrainer(model2)
            trainer2.compile_model()
            trainer2.train_model()

            logger.info("Pos Net training finished. Model weights have been saved.")
            logger.info("Shutting down tensorboard server.")
            tb_sup.finalize() 

        except ValueError as error:
            logger.error(str(error))
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
            logger.info("Testing neural network {option}".format(option=test_args.option))
        except ValueError as error:
            print(str(error))
            exit(1)

    def predict(self):
        pass