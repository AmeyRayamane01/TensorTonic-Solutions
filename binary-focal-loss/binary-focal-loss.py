import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    # Convert inputs to numpy arrays
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)

    # Compute p_t (probability of true class)
    pt = np.where(y == 1, p, 1 - p)

    # Compute focal loss
    loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)

    # Return mean loss
    return float(np.mean(loss))