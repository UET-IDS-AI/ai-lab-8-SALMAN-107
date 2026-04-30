import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_pass(X, W1, W2, W3):
    z1 = X @ W1
    h1 = sigmoid(z1)

    z2 = h1 @ W2
    h2 = sigmoid(z2)

    z3 = h2 @ W3
    y = sigmoid(z3)

    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    n = X.shape[0]

    y_clipped = np.clip(y, 1e-9, 1 - 1e-9)
    loss = -np.mean(label * np.log(y_clipped) + (1 - label) * np.log(1 - y_clipped))

    delta3 = y - label
    dW3 = (h2.T @ delta3) / n

    delta2 = (delta3 @ W3.T) * (h2 * (1 - h2))
    dW2 = (h1.T @ delta2) / n

    delta1 = (delta2 @ W2.T) * (h1 * (1 - h1))
    dW1 = (X.T @ delta1) / n

    return dW1, dW2, dW3, loss


if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 6) * 0.1
    W2 = np.random.randn(6, 5) * 0.1
    W3 = np.random.randn(5, 1) * 0.1
    labels = np.array([[1], [0], [1], [1], [0]], dtype=float)

    h1, h2, y = forward_pass(X, W1, W2, W3)
    print("h1 shape:", h1.shape)
    print("h2 shape:", h2.shape)
    print("y  shape:", y.shape)
    print("Predictions:\n", y)

    dW1, dW2, dW3, loss = backward_pass(X, h1, h2, y, labels, W1, W2, W3)
    print(f"\nLoss: {loss:.6f}")
    print("dW1 shape:", dW1.shape)
    print("dW2 shape:", dW2.shape)
    print("dW3 shape:", dW3.shape)
