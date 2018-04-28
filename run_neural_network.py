import sys
from save import (
    NeuralNetworkLoader,
    format_nn,
)


def run_test(argv):
    nn = NeuralNetworkLoader.load(argv[1])
    with open(argv[2], 'r') as f:
        data_input = map(float, f.readline().replace('\n', '').split(' '))
        data_output = map(float, f.readline().replace('\n', '').split(' '))
        dataset = (data_input, data_output)

    nn.learn(dataset)
    print format_nn(nn)


run_test(sys.argv)
