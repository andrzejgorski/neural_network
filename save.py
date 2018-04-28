import numpy as np


class Weights(object):
    endl = '\n'
    def __init__(self, weights=None):
        self._weights = weights or []

    def __getitem__(self, key):
        return self._weights[key]

    def save_weights(self, filename):
        with open(filename, 'w') as f:
            f.write(str(len(self._weights)) + self.endl)
            for weight in self._weights:
                f.write(
                    '{} {}'.format(len(weight), len(weight[0])) + self.endl)
                for row in weight:
                    f.write(' '.join(str(f) for f in row) + self.endl)

    def load_weights(self, filename):
        self._weights = []
        with open(filename, 'r') as f:
            layers = int(f.readline())
            for _ in range(layers):
                list_input = []
                rows, _ = (int(v) for v in f.readline().split(' '))
                for __ in range(rows):
                    new_row = np.array(
                        [float(numb) for numb in f.readline().split(' ')],
                        dtype=np.float64
                    )
                    list_input.append(new_row)

                self._weights.append(np.array(list_input))


def get_weights(nnl):
    return [layer.weights for layer in nnl.layers]
