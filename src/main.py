import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import spiral_data


# more of this can "describe" non-linear function
def relu_output(values):
    return np.maximum(0, values)


# non-negative (thanks to exp) and normalized (scores sums up to 1) confidence score
# first normalization (normalized_values) prevent big values to overflow when exp
# the more one of the confidence score is higher respect to others and more that final confidence score is near 1
def softmax_output(values):
    normalized_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return normalized_values / np.sum(normalized_values, axis=1, keepdims=True)


# each input is linked to each neuron/output (https://epynn.net/_images/Dense-01.svg)
def dense_layer(inputs_count, outputs_count):
    weights = .01 * np.random.randn(inputs_count, outputs_count)
    biases = np.zeros((1, outputs_count))

    return weights, biases


def dense_layer_output(inputs, weights, biases, activation_function=lambda x: x):
    return activation_function(np.dot(inputs, weights) + biases)


if __name__ == '__main__':
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)

    # print(first_layer_output[:5])

    # first_layer_output = dense_layer_output(X, *dense_layer(2, 3), softmax_output)
    # print(first_layer_output[:5])

    first_layer_output = dense_layer_output(X, *dense_layer(2, 3), relu_output)
    second_layer_output = dense_layer_output(first_layer_output, *dense_layer(3, 3), softmax_output)
    print(second_layer_output[: 5])

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
