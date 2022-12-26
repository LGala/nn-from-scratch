import nnfs
import numpy as np

from nnfs.datasets import spiral_data


# each input is linked to each neuron/output (https://epynn.net/_images/Dense-01.svg)
def dense_layer(inputs_count, outputs_count):
    weights = .01 * np.random.randn(inputs_count, outputs_count)
    biases = np.zeros((1, outputs_count))

    return weights, biases


def dense_layer_output(inputs, weights, biases, activation_function=lambda x: x):
    return activation_function(np.dot(inputs, weights) + biases)


# more of this can "describe" non-linear function
def relu_output(values):
    return np.maximum(0, values)


# non-negative and normalized (sum up to 1) confidence score
def softmax_output(values):
    # used to prevent big values to overflow and to make negative values positives
    normalized_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    # the following expression has 2 functions:
    #   - the samples in the resulting matrix will contain confidence scores that sum up to 1
    #   - more one of the confidence scores is higher respect to others and more is near 1
    return normalized_values / np.sum(normalized_values, axis=1, keepdims=True)


def categorical_cross_entropy_loss_function_mean(confidence_scores, actual_categories: np.ndarray):
    # in order to prevent log(0) (that will result in -inf) if some confidence scores == 0
    clipped_confidence_scores = np.clip(confidence_scores, 1e-7, 1 - 1e-7)
    # this is a simple math trick to simplify the loss function calculation, example:
    #   - confidences_scores = [[.1, .2, .7]],
    #   - actual_categories = [2] (that is a short way to write [[0, 0, 1]])
    #   - the loss function full formula should be -sum(log(.1) * 0 + log(.2) * 0 + log(.7) * 1) which is the same
    #     of -sum(log(.7) * 1) which is the same of using the actual_categories index to select the non-zero
    #     confidences_score, basically: -sum([[.1, .2, .7]][0][2]) which is the same of the below expression
    losses = - np.log(clipped_confidence_scores[[range(len(clipped_confidence_scores))], actual_categories])
    return np.mean(losses)


def accuracy(confidence_scores, actual_categories):
    predicted_class = np.argmax(confidence_scores, axis=1)
    return np.mean(predicted_class == actual_categories)


if __name__ == '__main__':
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)

    first_layer_output = dense_layer_output(X, *dense_layer(2, 3), relu_output)
    second_layer_output = dense_layer_output(first_layer_output, *dense_layer(3, 3), softmax_output)

    loss_mean = categorical_cross_entropy_loss_function_mean(second_layer_output, y)
    accuracy = accuracy(second_layer_output, y)

    print(accuracy)

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
