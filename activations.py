import numpy as np


class Activation_R:
    def fpass(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SM:
    def fpass(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob


