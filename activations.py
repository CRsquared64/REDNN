import numpy
import numpy as np


class Activation_R:
    def fpass(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SM:
