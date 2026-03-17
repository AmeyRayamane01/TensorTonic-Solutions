import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerically stable softmax
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)

    # Denominator of softmax
    denom = np.sum(exp_S, axis=1)

    # Positive similarities (diagonal)
    pos = np.diag(exp_S)

    # InfoNCE loss
    loss = -np.log(pos / denom)

    return float(np.mean(loss))