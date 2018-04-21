import numpy as np


endl = '\n'


def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    if not derivative:
        return np.exp(x) / float(sum(np.exp(x)))
    sm = x.reshape((-1, 1))
    return np.diag(x) - np.dot(sm, sm.T)


def ReLU(x, derivative=False):
    if not derivative:
        return np.array([max(0, y) for y in x])
    return np.array([1 if y > 0 else 0 for y in x])


def np_init():
    np.random.seed(1)


alpha = 0.1


# def _stack_ones(layer_input):
#     ones = np.ones((layer_input.shape[0], 1))
#     return np.hstack((ones, layer_input))


def _stack_ones(layer_input):
    ones = np.ones((layer_input.shape[0], 1))
    return np.hstack((layer_input, ones))


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
        print self.final_output
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


class Weights(object):
    def __init__(self, weights=None):
        self._weights = weights or []

    def __getitem__(self, key):
        return self._weights[key]

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

    def calc(self, dataset):
        self.calc = NeutralNetworkCalculation(self, dataset)
        return self.calc.run()


def create_simple_dataset():
    X = np.array([
        # [0, 0, 1],
        # [0, 1, 1],
        # [1, 0, 0],
        # [1, 1, 0],
        # [1, 0, 1],
        [1, 1, 1],
    ])

    # y = np.array([[0, 1, 0, 1, 1, 0]]).T
    y = np.array([[0]]).T
    return (X, y)



def sigmoid2(input_, output_, derivative=False):
    if not derivative:
        for i in range(len(input_)):
            output_[i] = 1.0 / (1 + np.exp(-input_[i]))
    else:
        for i in range(len(input_)):
            output_[i][i] = input_[i] * (1 - input_[i])


def id_(input_, output_, derivative=False):
    if not derivative:
        for i in range(len(input_)):
            output_[i] = input_[i]
    else:
        for i in range(len(input_)):
            output_[i][i] = input_[i] * (1 - input_[i])


def matrix_mult(first, second, out, first_r, second_c, first_c):
    for b in range(first_r):
        for c in range(second_c):
            out[b][c] = 0
            for a in range(first_c):
                out[b][c] += first[b][a] * second[a][c]


class Layer(object):
    def __init__(self, input_size, output_size, activation_function=None,
                 alpha=0.1, previous=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.activation_function = activation_function or sigmoid2
        self.previous = previous

        # input layer
        if previous:
            self.input_value = previous.output_value
            self.input_error = previous.output_error
        else:
            self.input_value = np.ones((input_size + 1, 1))
            self.input_error = np.ones((input_size + 1, 1))

        # mid layer
        self.mid_value = np.zeros((output_size, 1))
        self.mid_error = np.zeros((output_size, 1))

        # output layer
        self.output_value = np.ones((output_size + 1, 1))
        self.output_error = np.zeros((output_size, 1))

        # weights
        self.weights = np.ones((output_size, input_size + 1))
        self.gradient = np.ones((output_size, input_size + 1))

    def calc(self):
        matrix_mult(
            first=self.weights,
            second=self.input_value,
            out=self.mid_value,
            first_r=self.output_size,
            second_c=1,
            first_c=self.input_size+1
        )

        self.activation_function(self.mid_value, self.output_value)

    def backpropagation(self):
        pass


class NeutralNetworkLayered(object):
    def __init__(self, layers, act_func=None, last_act_func=None):
        act_func = act_func or sigmoid2
        last_act_func = last_act_func or id_

        layers_len = len(layers)
        self.layers = [Layer(
            layers[0],
            layers[1],
            activation_function=act_func
        )]
        for i in range(1, layers_len - 2):
            self.layers.append(Layer(
                layers[i],
                layers[i + 1],
                activation_function=act_func,
                previous=self.layers[i - 1]
            ))

        self.layers.append(Layer(
            layers[layers_len - 2],
            layers[layers_len - 1],
            activation_function=last_act_func,
            previous=self.layers[layers_len - 3]
        ))

    def feed_weights(self, weights):
        for weight, layer in zip(weights, self.layers):
            for i in range(len(weight)):
                for j in range(len(weight[0])):
                    layer.weights[j][i] = weight[i][j]

    def calc(self, dataset):
        for i in range(self.layers[0].input_size):
            self.layers[0].input_value[i][0] = dataset[0][0][i]
        for layer in self.layers:
            layer.calc()
        return self.layers[-1].output_value


nn = NeutralNetwork([3, 3, 1])
nn.load_weights('weights.txt')
dataset = create_simple_dataset()
nn.calc(dataset)

weights = Weights()
weights.load_weights('weights.txt')
nnl = NeutralNetworkLayered([3, 3, 1])
nnl.feed_weights(weights)
print(nnl.calc(dataset))
