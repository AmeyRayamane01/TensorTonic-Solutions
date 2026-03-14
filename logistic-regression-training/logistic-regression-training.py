import numpy as np

def train_logistic_regression(X, y, lr, steps):

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Handle 1D input like [1,2,3]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(int(steps)):

        z = np.dot(X, w) + b
        p = _sigmoid(z)

        error = p - y

        dw = np.dot(X.T, error) / n_samples
        db = np.sum(error) / n_samples

        w -= lr * dw
        b -= lr * db

    return w, b