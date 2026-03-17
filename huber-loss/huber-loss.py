import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # Piecewise loss
    quadratic = 0.5 * e**2
    linear = delta * (abs_e - 0.5 * delta)

    loss = np.where(abs_e <= delta, quadratic, linear)

    return float(np.mean(loss))