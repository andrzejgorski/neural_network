import numpy as np


class softmax(object):
    @classmethod
    def calc(cls, input_, output_):
        if not derivative:
            shifted = input_ - max(input_)
            result = np.exp(shifted) / float(sum(np.exp(shifted)))
            for i in range(len(output_)):
                output_[i] = result[i]

    @classmethod
    def derivative(cls, input_, output_):
        sm = input_.reshape((-1, 1))
        result = np.diag(input_) - np.dot(sm, sm.T)
        for i in range(len(output_)):
            for j in range(len(output_[0])):
                output_[i][j] = result[i][j]


class ReLU(object):
    @classmethod
    def calc(cls, input_, output_):
        result = np.array([max(0, y) for y in input_])
        for i in range(len(output_)):
            output_[i] = result[i]

    @classmethod
    def derivative(cls, x):
        result = np.array([1 if y > 0 else 0 for y in x])
        for i in range(len(output_)):
            output_[i][i] = result[i]


class sigmoid(object):
    name = 'sigmoid'
    @classmethod
    def calc(cls, input_, output_):
        for i in range(len(input_)):
            output_[i] = 1.0 / (1 + np.exp(-input_[i]))

    @classmethod
    def derivative(cls, input_, output_):
        for i in range(len(output_)):
            output_[i][i] = input_[i] * (1 - input_[i])


class id_(object):
    name = 'id'
    @classmethod
    def calc(cls, input_, output_):
        for i in range(len(input_)):
            output_[i] = input_[i]

    @classmethod
    def derivative(cls, input_, output_):
        for i in range(len(output_)):
            output_[i][i] = 1
