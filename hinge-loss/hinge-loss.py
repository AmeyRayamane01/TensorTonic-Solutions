import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    # Validate labels
    if not np.all((y_true == 1) | (y_true == -1)):
        raise ValueError("y_true must contain only -1 or +1")

    # Compute hinge loss
    loss = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        return loss