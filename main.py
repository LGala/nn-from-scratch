import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import spiral_data


def relu_output(layer_output):
    return np.maximum(0, layer_output)


def dense_layer(inputs_count, outputs_count):
    weights = .01 * np.random.randn(inputs_count, outputs_count)
    biases = np.zeros((1, outputs_count))

    return weights, biases


def dense_layer_output(inputs, weights, biases, activation_function=lambda x: x):
    return activation_function(np.dot(inputs, weights) + biases)


if __name__ == '__main__':
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    
    first_layer_output = dense_layer_output(X, *dense_layer(2, 3), relu_output)
    print(first_layer_output[:5])

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
