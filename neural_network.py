import numpy as np


endl = '\n'


def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def np_init():
    np.random.seed(1)


alpha = 0.1


def _stack_ones(layer_input):
    ones = np.ones((layer_input.shape[0], 1))
    return np.hstack((ones, layer_input))


class NeutralNetworkCalculation(object):
    def __init__(self, neutral_network, dataset):
        self.dataset = dataset
        self.neutral_network = neutral_network
        self._weights = neutral_network._weights
        self.layer_outputs = None

    def run(self):
        self.layer_outputs = [self.dataset[0]]
        for layer in range(len(self._weights)):
            stack_ones = _stack_ones(self.layer_outputs[layer])
            next_layer = np.dot(stack_ones, self._weights[layer])

            if layer == len(self._weights) - 1:
                self.final_output = next_layer

            self.layer_outputs.append(sigmoid(next_layer))
        return self.final_output


class BackPropagation(NeutralNetworkCalculation):
    def __init__(self, neutral_network, dataset):
        self.dataset = dataset
        self.neutral_network = neutral_network
        self._weights = neutral_network._weights
        self.layer_outputs = None

    def _calc_layer_errors(self, output_error):
        self._layer_errors = [output_error]
        last_error_layer = output_error
        for layer_ix in reversed(range(len(self.layer_outputs[: -1]))):
            sigmoid_derivative = sigmoid(
                self.layer_outputs[layer_ix], derivative=True)

            weights_derivative = np.dot(
                last_error_layer, self._weights[layer_ix].T[:, 1:])

            last_error_layer = sigmoid_derivative * weights_derivative
            self._layer_errors.append(last_error_layer)

        self._layer_errors.reverse()

    def _calc_gradients(self):
        self.gradients = []
        for index in range(len(self.layer_outputs) - 1):
            formatted_output = _stack_ones(self.layer_outputs[index])[:, :, np.newaxis]
            formatted_error = self._layer_errors[index + 1][: , np.newaxis, :]
            partial_derivatives = formatted_output * formatted_error
            total_gradient = np.average(partial_derivatives, axis=0)
            self.gradients.append(total_gradient)

    def run(self):
        output_error = super(BackPropagation, self).run() - self.dataset[1]
        self._calc_layer_errors(output_error)
        self._calc_gradients()


class NeutralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def save_weights(self, filename):
        with open(filename, 'w') as f:
            f.write(str(len(self._weights)) + endl)
            for weight in self._weights:
                f.write('{} {}'.format(len(weight), len(weight[0])) + endl)
                for row in weight:
                    f.write(' '.join(str(f) for f in row) + endl)

    def load_weights(self, filename):
        self._weights = []
        with open(filename, 'r') as f:
            layers = int(f.readline())
            for _ in range(layers):
                list_input = []
                rows, columns = (int(v) for v in f.readline().split(' '))
                for __ in range(rows):
                    new_row = np.array(
                        [float(numb) for numb in  f.readline().split(' ')],
                        dtype=np.float64
                    )
                    list_input.append(new_row)

                self._weights.append(np.array(list_input))

    def create_weights(self):
        self._weights = []
        for inn, out in zip(self.layers[:-1], self.layers[1:]):
            self._weights.append(2 * np.random.random((inn + 1, out)) - 1)

    def _T_without_first_col(self, array):
        return array.T[:, 1: ]

    def learn(self, dataset):
        backp = BackPropagation(self, dataset)
        backp.run()
        for index in range(len(self._weights)):
            self._weights[index] -= alpha * backp.gradients[index]


def create_simple_dataset():
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    y = np.array([[0, 1, 0, 1, 1, 0]]).T
    return (X, y)


nn = NeutralNetwork([3, 3, 1])
nn.load_weights('weights.txt')
nn.learn(create_simple_dataset())
# nn.save_weights('w2.txt')
