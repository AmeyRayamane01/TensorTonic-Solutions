import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    # Ensure batch dimension
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        y = y.reshape(1)

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)

    # Loss components
    pos_loss = y * (d ** 2)
    neg_loss = (1 - y) * np.maximum(0, margin - d) ** 2

    loss = pos_loss + neg_loss

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        return loss