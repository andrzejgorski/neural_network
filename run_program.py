#!/usr/bin/env python
import sys
import argparse
from prepare_files import FilePreparer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        help="training data path",
        default=None,
        dest='training_data',
    )
    parser.add_argument(
        "--epsilon",
        help="treshhold error value"
    )
    parser.add_argument(
        "--learnig_rate",
        help="learning rate for backpropagation",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--epochs",
        help="maximum number of epochs"
    )
    parser.add_argument(
        "--random",
        help="if true, initial weights are set randomly; if false, initial weights are set to 1",
        type=bool,
        default=True
    )
    args = parser.parse_args(sys.argv[1:])
    return args


def run_program():
    args = parse_args()
    if args.training_data is not None:
        file_preparer = FilePreparer(args.training_data, 'inputs')
        file_preparer.prepare_files()
    # RUN PROGRAM
    # result = subprocess.check_output(program + (nn_file,) + (in_file,))


if __name__ == "__main__":
    run_program()
