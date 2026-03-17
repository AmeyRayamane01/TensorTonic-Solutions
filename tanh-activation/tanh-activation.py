import numpy as np

def tanh(x):
    x = np.asarray(x, dtype=float)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))