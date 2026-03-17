import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    p = np.asarray(predictions, dtype=float)
    K = p.shape[0]

    # Build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # Cross-entropy loss
    loss = -np.sum(q * np.log(p))

    return float(loss)