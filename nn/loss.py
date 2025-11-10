import numpy as np

def mse(y, pred):
    return np.mean(np.power(y - pred, 2))

def mse_diff(y, pred):
    return 2 * (pred - y) / np.size(y)

def binary_cross_entropy(y, pred):
    return np.mean(-y * np.log(pred) - (1 - y) * np.log(1 - pred))

def binary_cross_entropy_diff(y, pred):
    return ((1 - y) / (1 - pred) - y / pred) / np.size(y)

def cross_entropy(y, pred):
    return -np.mean(np.sum(y * np.log(pred), axis=1))

def cross_entropy_diff(y, pred):
    return (pred - y) / y.shape[0]


    
