#!/usr/bin/python3
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        help="training data path",
        required=True,
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
        help="maximum number of epochs",
        type=bool,
        default=True
    )
    return parser.parse_args()


def run_program():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    run_program()
