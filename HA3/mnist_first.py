import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml


# Load and preprocess the MNIST dataset
def load_mnist_3_8():
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.astype(float) / 255.0  # Normalize pixel values
    y = mnist.target.astype(int)

    # Filter for digits 3 and 8
    mask = (y == 3) | (y == 8)
    X = X[mask]
    y = y[mask]

    # Convert labels: 3 -> 0, 8 -> 1
    y = (y == 8).astype(int)

    # Add bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X, y


# Helper functions for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))  # Clip to prevent overflow


def compute_loss(X, y, theta, mu=1.0):
    m = len(y)
    h = sigmoid(X @ theta)

    # Compute the cross-entropy loss
    epsilon = 1e-15  # Small value to prevent log(0)
    h = np.clip(h, epsilon, 1 - epsilon)
    cross_entropy = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    # Add L2 regularization term (excluding bias term)
    reg_term = (mu / (2 * m)) * np.sum(theta[1:] ** 2)

    return cross_entropy + reg_term


def compute_gradient(X, y, theta, mu=1.0):
    m = len(y)
    h = sigmoid(X @ theta)

    # Compute gradient of cross-entropy loss
    grad = (1 / m) * X.T @ (h - y)

    # Add L2 regularization term (excluding bias term)
    reg_term = np.zeros_like(theta)
    reg_term[1:] = (mu / m) * theta[1:]

    return grad + reg_term


# Gradient Descent algorithm
def gradient_descent(X, y, num_iterations=100, learning_rate=0.001, mu=1.0):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    iterations = []

    for i in range(num_iterations + 1):  # +1 to include iteration 0
        # Compute gradient
        grad = compute_gradient(X, y, theta, mu)

        # Update parameters
        theta = theta - learning_rate * grad

        # Record loss every 10 iterations
        if i % 10 == 0:
            loss = compute_loss(X, y, theta, mu)
            losses.append(loss)
            iterations.append(i)
            print(f"GD Iteration {i}, Loss: {loss:.6f}")

    return theta, losses, iterations


# Mini-batch SGD algorithm
def mini_batch_sgd(
    X, y, num_iterations=100, batch_size=10, learning_rate=0.001, mu=1.0
):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    iterations = []

    for i in range(num_iterations + 1):  # +1 to include iteration 0
        if i > 0:  # Skip update for iteration 0 to record initial loss
            # Sample random mini-batch
            indices = np.random.choice(m, batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # Compute gradient on mini-batch
            grad = compute_gradient(X_batch, y_batch, theta, mu)

            # Update parameters
            theta = theta - learning_rate * grad

        # Record loss every 10 iterations (computed on full dataset)
        if i % 10 == 0:
            loss = compute_loss(X, y, theta, mu)
            losses.append(loss)
            iterations.append(i)
            print(f"SGD Iteration {i}, Loss: {loss:.6f}")

    return theta, losses, iterations


# Main execution
def main():
    # Load data
    X, y = load_mnist_3_8()
    print(f"Dataset shape: {X.shape}, Labels distribution: {np.bincount(y)}")

    # Run Gradient Descent
    print("\nRunning Gradient Descent...")
    start_time = time.time()
    theta_gd, losses_gd, iter_gd = gradient_descent(
        X, y, num_iterations=100, learning_rate=0.001, mu=1.0
    )
    gd_time = time.time() - start_time
    print(f"GD completed in {gd_time:.2f} seconds")

    # Run Mini-batch SGD
    print("\nRunning Mini-batch SGD...")
    start_time = time.time()
    theta_sgd, losses_sgd, iter_sgd = mini_batch_sgd(
        X, y, num_iterations=100, batch_size=10, learning_rate=0.001, mu=1.0
    )
    sgd_time = time.time() - start_time
    print(f"Mini-batch SGD completed in {sgd_time:.2f} seconds")

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(iter_gd, losses_gd, "b-o", linewidth=2, label=f"GD (time: {gd_time:.2f}s)")
    plt.plot(
        iter_sgd,
        losses_sgd,
        "r-o",
        linewidth=2,
        label=f"Mini-batch SGD (time: {sgd_time:.2f}s)",
    )

    plt.title("Training Loss vs. Iterations for Digit 3-8 Classification")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_comparison.png")
    plt.show()

    # Calculate and print accuracies
    y_pred_gd = (sigmoid(X @ theta_gd) >= 0.5).astype(int)
    accuracy_gd = np.mean(y_pred_gd == y)

    y_pred_sgd = (sigmoid(X @ theta_sgd) >= 0.5).astype(int)
    accuracy_sgd = np.mean(y_pred_sgd == y)

    print(f"\nAccuracy with GD: {accuracy_gd:.4f}")
    print(f"Accuracy with Mini-batch SGD: {accuracy_sgd:.4f}")


if __name__ == "__main__":
    main()
