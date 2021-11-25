import argparse
import sys
import os

from multiprocessing import Process
from uuid import uuid4

from app.resources.constants import (
    COMMAND_LINE_OPTIONS, 
    COMMAND_LINE_DESCRIPTION, 
    CMD_TRAIN_DESCRIPTION
)
from app.services.trainer import Trainer
from app.services.preprocessor import PreProcessor
from app.nn_models.nn_orient import ReOrientNet
from app.models.options import Option
from app.utils.process_workflow import do_preprocessing_work

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
        """Preprocess will create the csv files with multithreading, 
        one process is spawned for every file, so there will be 4 processes 
        in total.  
        """
        
        print("Running training set initializers...\n")

        parser = argparse.ArgumentParser(
            description = CMD_TRAIN_DESCRIPTION,
        )

        vars(parser.parse_args(sys.argv[2:]))
        
        print("Creating train and test csvs...")

        train_file_path = "datasets/csvs/train.csv"
        test_file_path1 = "datasets/csvs/building1.csv"
        test_file_path2 = "datasets/csvs/building2.csv"
        test_file_path3 = "datasets/csvs/building3.csv"

        if not os.path.exists(train_file_path):
            option = Option.TRAIN
            file_path = "datasets/csvs/train.csv"

            process = Process(target=do_preprocessing_work, args=(uuid4(), file_path, option))
            process.start()
  
        if not os.path.exists(test_file_path1):
            option = Option.BLD1
            file_path = "datasets/csvs/building1.csv"

            process = Process(target=do_preprocessing_work, args=(uuid4(), file_path, option))
            process.start()

        if not os.path.exists(test_file_path2):
            option = Option.BLD2
            file_path = "datasets/csvs/building2.csv"

            process = Process(target=do_preprocessing_work, args=(uuid4(), file_path, option))
            process.start()

        if not os.path.exists(test_file_path3):
            option = Option.BLD3
            file_path = "datasets/csvs/building3.csv"

            process = Process(target=do_preprocessing_work, args=(uuid4(), file_path, option))
            process.start()





    def train(self):
        pass
        # pre = PreProcessor(train_csv=train_file_path, test_csv=test_file_path)
        # model = ReOrientNet()
        # trainer = Trainer(model)

        # print("Preprocessing data....")

        # (X_train, y_train) = pre.reshape_data(is_train = True)
        # (X_test, y_test) = pre.reshape_data(is_train = False)

        # print("Running nn training, create .env file to change hyper parameters.")

        # trainer.compile_model()
        # trainer.train_model(X_train, y_train)


    def predict(self):
        pass