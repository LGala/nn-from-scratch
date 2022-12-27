import nnfs
import numpy as np

from nnfs.datasets import spiral_data


class DenseLayer:
    def __init__(self, inputs_count, outputs_count):
        self.output = None
        self.weights = .01 * np.random.randn(inputs_count, outputs_count)
        self.biases = np.zeros((1, outputs_count))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# more of this can "describe" non-linear functions
def relu(values):
    return np.maximum(0, values)


# non-negative and normalized (sum up to 1) confidence score
def softmax(values):
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

    layer_1, layer_2 = DenseLayer(2, 3), DenseLayer(3, 3)

    layer_1.forward(X)
    relu_1_output = relu(layer_1.output)

    layer_2.forward(relu_1_output)
    softmax_1_output = softmax(layer_2.output)

    loss_mean = categorical_cross_entropy_loss_function_mean(softmax_1_output, y)
    accuracy = accuracy(softmax_1_output, y)

    print(accuracy)

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
