import numpy as np
import cv2

class LayerD:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def fpass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

