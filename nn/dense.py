import numpy as np
from .layers import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, lr=None):
        self.weights_gradient = np.dot(output_gradient, self.input.T)
        self.bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        return np.dot(self.weights.T, output_gradient)
