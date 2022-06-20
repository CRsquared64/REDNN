import numpy as np
import cv2


class LayerD:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def fpass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Loss:
    def calc(self, output, y):
        s_losses = self.fpass(output, y)
        d_loss = np.mean(s_losses)
        return d_loss

class Loss_entropy(Loss):
    def fpass(self, y_pre, y_tru):
        samples = len(y_pre)
        y_pre_clipped = np.clip(y_pre, 1e-7, 1-1e-7)

        if len(y_tru.shape) == 1:
            c_confidences = y_pre_clipped[range(samples), y_tru]
        elif len(y_tru.shape) == 2:
            c_confidences = np.sum(y_pre_clipped*y_tru, axis=1)