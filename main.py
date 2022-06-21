import numpy as np
from PIL import Image


from activations import Activation_R, Activation_SM


colour_im = Image.open('dataset/colour/istockphoto-1045886560-612x612.jpg')
colour_image_array = np.array(colour_im)

g_im = Image.open('dataset/greyscale/download.png')
g_image_array = np.array(g_im)
class LayerD:
    def __init__(self, n_inputs, n_neurons, inputs):
        self.inputs = inputs
        self.weights =  0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def fpass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def bpass(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Loss:
    def calc(self, output, y):
        s_losses = self.fpass(output, y)
        d_loss = np.mean(s_losses)
        return d_loss


class Loss_entropy(Loss):
    def fpass(self, y_pre, y_tru):
        samples = len(y_pre)
        y_pre_clipped = np.clip(y_pre, 1e-7, 1 - 1e-7)

        #c_confidences = y_pre_clipped[range(samples), y_tru] // 1 dimensional
        c_confidences = np.sum(y_pre_clipped * y_tru, axis=1)

        nlog_like = -np.log(c_confidences)
        return nlog_like

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


X = g_image_array
y = colour_image_array

layer1 = LayerD(2, 64)
active1 = Activation_R()

layer2 = LayerD(64, 3)
active2 = Activation_SM()

layer1.fpass(X)
active1.fpass(layer1.output)

layer2.fpass(active1.output)
active2.fpass(layer2.output)

print(active2.output)
loss_func = Loss_entropy()
loss = loss_func.calc(active2.output, y)

print("Loss:", loss)