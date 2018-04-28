from collections import deque
import numpy as np
from activation_functions import id_, sigmoid


def matrix_mult(first, second, out, first_r, second_c, first_c,
                t_first=False, t_second=False, t_out=False,
                shift_second=False):
    for b in range(first_r):
        for c in range(second_c):
            if not t_out:
                out[b][c] = 0
            else:
                out[c][b] = 0

            for a in range(first_c):
                if not t_first:
                    ff = first[b][a]
                else:
                    ff = first[a][b]

                if shift_second:
                    second_c_iterator = c + 1
                else:
                    second_c_iterator = c

                if not t_second:
                    ss = second[a][second_c_iterator]
                else:
                    ss = second[second_c_iterator][a]

                if not t_out:
                    out[b][c] += ff * ss
                else:
                    out[c][b] += ff * ss


class Layer(object):
    def __init__(self, input_size, output_size, activation_function=None,
                 alpha=0.1, previous=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.activation_function = activation_function or sigmoid
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

    def calc_errors(self):
        derivatives = np.zeros((self.output_size, self.output_size))
        self.activation_function(
            self.output_value, derivatives, derivative=True)

        matrix_mult(
            first=derivatives,
            second=self.output_error,
            out=self.mid_error,
            first_r=self.output_size,
            second_c=1,
            first_c=self.output_size
        )

        matrix_mult(
            first=self.mid_error,
            second=self.weights,
            out=self.input_error,
            first_r=1,
            second_c=self.input_size,
            first_c=self.output_size,
            t_first=True,
            t_out=True,
            shift_second=True
        )

    def calc_gradient(self):
        matrix_mult(
            first=self.mid_error,
            second=self.input_value,
            out=self.gradient,
            first_r=self.output_size,
            second_c=self.input_size + 1,
            first_c=1,
            t_second=True
        )

    def update_weights(self):
        for i in range(self.output_size):
            for j in range(self.input_size + 1):
                self.weights[i][j] -= self.alpha * self.gradient[i][j]

    def feed_output(self, proper_output):
        for i in range(self.output_size):
            self.output_error[i] = self.output_value[i] - proper_output[0][i]


class NeuralNetworkLayered(object):
    def __init__(self, layers, act_func=None, last_act_func=None):
        act_func = act_func or sigmoid
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

    def feed_weights(self, weights, transpose=False):
        for weight, layer in zip(weights, self.layers):
            for i in range(len(weight)):
                for j in range(len(weight[0])):
                    if transpose:
                        layer.weights[j][i] = weight[i][j]
                    else:
                        layer.weights[i][j] = weight[i][j]

    def learn(self, dataset):
        output = self.calc(dataset)
        self.layers[-1].feed_output(dataset[1])
        for layer in reversed(self.layers):
            layer.calc_errors()
        for layer in reversed(self.layers):
            layer.calc_gradient()
            layer.update_weights()

    def calc(self, dataset):
        for i in range(self.layers[0].input_size):
            self.layers[0].input_value[i][0] = dataset[0][0][i]
        for layer in self.layers:
            layer.calc()
        return self.layers[-1].output_value

    def get_weights(self):
        return [layer.weights for layer in self.layers]


# Move to tests
def check_equal(arr1, arr2):
    for row1, row2 in zip(arr1, arr2):
        for cell1, cell2 in zip(row1, row2):
            assert abs(cell1 - cell2) < 0.000001
