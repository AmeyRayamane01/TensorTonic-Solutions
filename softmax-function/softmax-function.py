import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    elif x.ndim == 2:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)