

def softmax(x, derivative=False):
    if not derivative:
        shifted = x - max(x)
        return np.exp(shifted) / float(sum(np.exp(shifted)))
    sm = x.reshape((-1, 1))
    return np.diag(x) - np.dot(sm, sm.T)


def sigmoid(input_, output_, derivative=False):
    if not derivative:
        for i in range(len(input_)):
            output_[i] = 1.0 / (1 + np.exp(-input_[i]))
    else:
        for i in range(len(output_)):
            output_[i][i] = input_[i] * (1 - input_[i])


def id_(input_, output_, derivative=False):
    if not derivative:
        for i in range(len(input_)):
            output_[i] = input_[i]
    else:
        for i in range(len(output_)):
            output_[i][i] = 1


def ReLU(x, derivative=False):
    if not derivative:
        return np.array([max(0, y) for y in x])
    return np.array([1 if y > 0 else 0 for y in x])
