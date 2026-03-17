import numpy as np

def swish(x):
    x = np.asarray(x, dtype=float)

    # numerically stable sigmoid
    sigmoid = 1 / (1 + np.exp(-x))

    return x * sigmoid