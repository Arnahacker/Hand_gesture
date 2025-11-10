import numpy as np

class NeuralNetwork:
    def __init__(self, layers, loss, loss_prime):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient

    def fit(self, x_train, y_train, epochs, optimizer=None, verbose=True):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):

                x = np.reshape(x, (x.shape[0], 1))
                y = np.reshape(y, (y.shape[0], 1))

                output = self.forward(x)

                error += self.loss(y, output)

                grad = self.loss_prime(y, output)

                self.backward(grad, optimizer.lr if optimizer else 0.01)

                if optimizer:
                    optimizer.step(self.layers)

            error /= len(x_train)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {error:.6f}")

    def predict(self, x_data):
        results = []
        for x in x_data:
            x = np.reshape(x, (x.shape[0], 1))
            results.append(self.forward(x))
        return results

    def evaluate(self, x_test, y_test):
        loss_sum = 0
        for x, y in zip(x_test, y_test):
            x = np.reshape(x, (x.shape[0], 1))
            y = np.reshape(y, (y.shape[0], 1))
            output = self.forward(x)
            loss_sum += self.loss(y, output)
        return loss_sum / len(x_test)

