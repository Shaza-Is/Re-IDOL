import argparse
import sys
import os

from loguru import logger

from app.resources.constants import (
    COMMAND_LINE_OPTIONS, 
    COMMAND_LINE_DESCRIPTION, 
    CMD_TRAIN_DESCRIPTION,
    CMD_TEST_DESCRIPTION
)
from app.services.orient_trainer import OrientTrainer
from app.services.pos_trainer import PosTrainer
from app.services.tensorboard_process import TensorboardSupervisor
from app.models.options import Option
from app.models.cmd_args import CommonArgs 
from app.utils.initializers import initialize_data, get_latest_checkpoint

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
        tb_sup = TensorboardSupervisor(tensorboard_log_path)


        try:
            train_args = CommonArgs.parse_obj(args)
            logger.info("Attempting to train ReOrient Net on Building {option}".format(option=train_args.option))
            logger.info("Starting Tensorboard server at http://localhost:6006")

            df = initialize_data(train_args.option)

            latest_checkpoint = get_latest_checkpoint("orient", train_args.option)
            initial_epoch = 0

            if len(latest_checkpoint) > 0:

                if latest_checkpoint[1] > 0: 
                    initial_epoch = latest_checkpoint[1]
            

            trainer = OrientTrainer(train_args.option, df, is_reduced=True)
            trainer.compile_model(latest_checkpoint=latest_checkpoint_orient)
            trainer.display_model()
            trainer.train_model(initial_epoch=initial_epoch_orient)
            results = trainer.predict()

            logger.info("ReOrient Net training finished. Model has been saved.")
            logger.info("Attempting to train Pos Net {option}".format(option=train_args.option))
            
            latest_checkpoint_pos = get_latest_checkpoint("pos", train_args.option)
            initial_epoch_pos = 0

            if latest_checkpoint_pos and latest_checkpoint_pos[1] > 0: 
                initial_epoch_pos = latest_checkpoint_pos[1]

            trainer2 = PosTrainer(train_args.option, df, results, is_reduced=True)
            trainer2.compile_model(latest_checkpoint=latest_checkpoint_pos)
            trainer2.display_model()
            trainer2.train_model(initial_epoch=initial_epoch_pos)

            logger.info("Pos Net training finished. Model weights have been saved.")
            logger.info("Shutting down tensorboard server.")
            tb_sup.finalize() 

        except ValueError as error:
            logger.error(str(error))
            tb_sup.finalize()
            exit(1)


    def test(self):
        """test is used to perform testing with one of the neural networks
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
        tensorboard_log_path = "logs/evaluate"
        tb_sup = TensorboardSupervisor(tensorboard_log_path)

        try:
            test_args = CommonArgs.parse_obj(args)
            
            if not os.path.exists("saves/orient/"):
                logger.error("No saved models, cannot test model. Please train model before calling test function.")
                exit(1)

            logger.info("Testing neural network on building {option}".format(option=test_args.option))

            df = initialize_data(test_args.option, is_training = False)

            trainer = OrientTrainer(test_args.option, df, is_reduced = True)
            trainer.evaluate_model()
            tb_sup.finalize()

        except ValueError as error:
            logger.error(str(error))
            tb_sup.finalize()
            exit(1)