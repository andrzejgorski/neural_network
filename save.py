import numpy as np
from neural_network import NeuralNetworkLayered


def format_nn(nnl):
    weights = nnl.get_weights()
    # result = [nnl.act_func.name + ' ' + nnl.last_act_func.name]
    result = []
    result.append(str(len(weights)))
    for weight in weights:
        result.append(
            '{} {}'.format(len(weight), len(weight[0])))
        for row in weight:
            result.append(' '.join(str(f) for f in row))
    return '\n'.join(result)


class NeuralNetworkLoader(object):
    endl = '\n'

    @classmethod
    def save(cls, nnl, filename):
        with open(filename, 'w') as f:
            f.write(nnl)

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
