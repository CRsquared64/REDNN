import numpy as np
import cv2
import nnfs
from nnfs.datasets import spiral_data

from activations import Activation_R, Activation_SM

nnfs.init()


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
        y_pre_clipped = np.clip(y_pre, 1e-7, 1 - 1e-7)

        if len(y_tru.shape) == 1:
            c_confidences = y_pre_clipped[range(samples), y_tru]
        elif len(y_tru.shape) == 2:
            c_confidences = np.sum(y_pre_clipped * y_tru, axis=1)

        nlog_like = -np.log(c_confidences)
        return nlog_like


X, y = spiral_data(samples=100, classes=3)

layer1 = LayerD(2, 3)
active1 = Activation_R()

layer2 = LayerD(3, 3)
active2 = Activation_SM()

layer1.fpass(X)
active1.fpass(layer1.output)

layer2.fpass(active1.output)
active2.fpass(layer2.output)

print(active2.output[:5])
loss_func = Loss_entropy()
loss = loss_func.calc()
