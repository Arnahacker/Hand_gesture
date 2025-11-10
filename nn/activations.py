import numpy as np
from .layers import Layer

class Activation(Layer):
    def __init__(self, activation, activation_diff):
        self.activation = activation
        self.activation_prime = activation_diff

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, *args, **kwargs):
        return np.multiply(output_gradient, self.activation_prime(self.input))
