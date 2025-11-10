import numpy as np
from .layers import Layer
from .activations import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):  
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=0, keepdims=True))
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        s = self.output  
        if s.ndim == 2 and s.shape[1] == 1:
            jacobian = np.diagflat(s.flatten()) - np.outer(s.flatten(), s.flatten())
            return np.dot(jacobian, output_gradient).reshape(s.shape)
        grads = []
        for i in range(s.shape[1]):
            si = s[:, i:i+1]
            jac = np.diagflat(si.flatten()) - np.outer(si.flatten(), si.flatten())
            grads.append(np.dot(jac, output_gradient[:, i:i+1]))
        return np.hstack(grads)