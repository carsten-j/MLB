import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Load and preprocess MNIST dataset
def load_mnist_3_8():
    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Filter only digits 3 and 8
    train_mask = (train_dataset.targets == 3) | (train_dataset.targets == 8)
    test_mask = (test_dataset.targets == 3) | (test_dataset.targets == 8)

    train_data = train_dataset.data[train_mask]
    train_targets = train_dataset.targets[train_mask]
    test_data = test_dataset.data[test_mask]
    test_targets = test_dataset.targets[test_mask]

    # Convert labels: 3 -> 0, 8 -> 1
    train_targets = (train_targets == 8).float()
    test_targets = (test_targets == 8).float()

    # Flatten and normalize images
    train_data = train_data.float().view(-1, 28 * 28) / 255.0
    test_data = test_data.float().view(-1, 28 * 28) / 255.0

    return train_data, train_targets, test_data, test_targets


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Define loss function with L2 regularization
def loss_function(model, outputs, targets, mu=1.0):
    # Binary cross-entropy loss
    bce_loss = nn.BCELoss()(outputs, targets.view(-1, 1))

    # L2 regularization term
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2) ** 2

    # Combined loss
    return bce_loss + mu * l2_reg / 2


# Gradient Descent optimizer
def gradient_descent(
    model, train_data, train_targets, learning_rate=0.001, iterations=100
):
    losses = []

    for i in range(iterations):
        # Forward pass
        outputs = model(train_data)

        # Calculate loss
        loss = loss_function(model, outputs, train_targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # Record loss every 10 iterations
        if (i + 1) % 10 == 0:
            losses.append(loss.item())
            print(f"GD Iteration {i + 1}, Loss: {loss.item()}")

    return losses


# Mini-batch SGD optimizer
def minibatch_sgd(
    model, train_data, train_targets, batch_size=10, learning_rate=0.001, iterations=100
):
    losses = []
    num_samples = len(train_data)

    for i in range(iterations):
        # Randomly sample a mini-batch
        indices = torch.randperm(num_samples)[:batch_size]
        batch_data = train_data[indices]
        batch_targets = train_targets[indices]

        # Forward pass
        outputs = model(batch_data)

        # Calculate loss on batch (for optimization)
        loss = loss_function(model, outputs, batch_targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # Record loss every 10 iterations (on full dataset for fair comparison)
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                full_outputs = model(train_data)
                full_loss = loss_function(model, full_outputs, train_targets)
                losses.append(full_loss.item())
                print(f"SGD Iteration {i + 1}, Loss: {full_loss.item()}")

    return losses


# Plot the training losses
def plot_losses(gd_losses, sgd_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(10, 10 * len(gd_losses) + 1, 10), gd_losses, label="Gradient Descent"
    )
    plt.plot(
        range(10, 10 * len(sgd_losses) + 1, 10), sgd_losses, label="Mini-batch SGD"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to run the experiment
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    iterations = 2000

    # Load data
    train_data, train_targets, test_data, test_targets = load_mnist_3_8()

    print(f"Number of training samples: {len(train_data)}")

    # Train with Gradient Descent
    print("Training with Gradient Descent...")
    gd_model = LogisticRegression()
    gd_losses = gradient_descent(
        gd_model, train_data, train_targets, learning_rate=0.001, iterations=iterations
    )

    # Train with Mini-batch SGD
    print("\nTraining with Mini-batch SGD...")
    sgd_model = LogisticRegression()
    sgd_losses = minibatch_sgd(
        sgd_model,
        train_data,
        train_targets,
        batch_size=10,
        learning_rate=0.001,
        iterations=iterations,
    )

    # Plot the training losses
    plot_losses(gd_losses, sgd_losses)


if __name__ == "__main__":
    main()
