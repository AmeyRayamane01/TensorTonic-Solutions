import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Validate shapes
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")

    # Compute focal loss
    loss = -(1 - p)**gamma * y * np.log(p) - (p**gamma) * (1 - y) * np.log(1 - p)

    # Mean loss
    return float(np.mean(loss))