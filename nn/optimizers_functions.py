import numpy as np
from .optimizers import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "weights") or not hasattr(layer, "bias"):
                continue
            layer.weights -= self.lr * layer.weights_gradient
            layer.bias -= self.lr * layer.bias_gradient


class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity_w = {}
        self.velocity_b = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "weights") or not hasattr(layer, "bias"):
                continue 

            if layer not in self.velocity_w:
                self.velocity_w[layer] = np.zeros_like(layer.weights)
                self.velocity_b[layer] = np.zeros_like(layer.bias)

            self.velocity_w[layer] = (
                self.momentum * self.velocity_w[layer]
                - self.lr * layer.weights_gradient
            )
            self.velocity_b[layer] = (
                self.momentum * self.velocity_b[layer]
                - self.lr * layer.bias_gradient
            )

            layer.weights += self.velocity_w[layer]
            layer.bias += self.velocity_b[layer]


class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = epsilon
        self.sq_grad = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "weights") or not hasattr(layer, "bias"):
                continue

            if layer not in self.sq_grad:
                self.sq_grad[layer] = [
                    np.zeros_like(layer.weights_gradient),
                    np.zeros_like(layer.bias_gradient),
                ]

            params = [layer.weights, layer.bias]
            grads = [layer.weights_gradient, layer.bias_gradient]

            for i in range(len(params)):
                self.sq_grad[layer][i] = (
                    self.beta * self.sq_grad[layer][i] + (1 - self.beta) * grads[i] ** 2
                )
                params[i] -= self.lr * grads[i] / (np.sqrt(self.sq_grad[layer][i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "weights") or not hasattr(layer, "bias"):
                continue

            if layer not in self.m:
                self.m[layer] = [
                    np.zeros_like(layer.weights_gradient),
                    np.zeros_like(layer.bias_gradient),
                ]
                self.v[layer] = [
                    np.zeros_like(layer.weights_gradient),
                    np.zeros_like(layer.bias_gradient),
                ]
                self.t[layer] = 0

            params = [layer.weights, layer.bias]
            grads = [layer.weights_gradient, layer.bias_gradient]
            self.t[layer] += 1

            for i in range(len(params)):
                self.m[layer][i] = self.beta1 * self.m[layer][i] + (1 - self.beta1) * grads[i]
                self.v[layer][i] = self.beta2 * self.v[layer][i] + (1 - self.beta2) * (grads[i] ** 2)

                m_hat = self.m[layer][i] / (1 - self.beta1 ** self.t[layer])
                v_hat = self.v[layer][i] / (1 - self.beta2 ** self.t[layer])

                params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
