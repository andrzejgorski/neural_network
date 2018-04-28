import numpy as np
from neural_network import NeuralNetworkLayered


class NeuralNetworkLoader(object):
    endl = '\n'

    @classmethod
    def save(cls, nnl, filename):
        weights = nnl.get_weights()
        with open(filename, 'w') as f:
            f.write(str(len(weights)) + cls.endl)
            for weight in weights:
                f.write(
                    '{} {}'.format(len(weight), len(weight[0])) + cls.endl)
                for row in weight:
                    f.write(' '.join(str(f) for f in row) + cls.endl)

    @classmethod
    def load(cls, filename):
        layers = []
        weights = []
        with open(filename, 'r') as f:
            layers_num = int(f.readline())
            for _ in range(layers_num):
                list_input = []
                rows, columns = (int(v) for v in f.readline().split(' '))
                if not layers:
                    layers.append(columns - 1)
                layers.append(rows)
                for __ in range(rows):
                    new_row = np.array(
                        [float(numb) for numb in f.readline().split(' ')],
                        dtype=np.float64
                    )
                    list_input.append(new_row)

                weights.append(np.array(list_input))

        nn = NeuralNetworkLayered(layers)
        nn.feed_weights(weights)
        return nn
