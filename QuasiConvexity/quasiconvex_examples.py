import matplotlib.pyplot as plt
import numpy as np

# Set up plotting parameters
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = [10, 6]


# Example 1: A simple 1D quasiconvex function
def example1(x):
    """A simple quasiconvex function: f(x) = |x|"""
    return np.abs(x)


# Example 2: A quasiconvex function that is not convex
def example2(x):
    """A quasiconvex function that is not convex: f(x) = 1/(1+x^2)"""
    return -1 / (1 + x**2)


# Example 3: A 2D quasiconvex function
def example3(x, y):
    """A 2D quasiconvex function: f(x,y) = max(|x|, |y|)"""
    return np.maximum(np.abs(x), np.abs(y))


# Example 4: Another 2D quasiconvex function that is not convex
def example4(x, y):
    """A 2D quasiconvex function that is not convex: f(x,y) = x^2 + y^2 / (1 + x^2 + y^2)"""
    return (x**2 + y**2) / (1 + x**2 + y**2)


# Example 5: Ceiling function (quasiconvex but not convex)
def example5(x):
    """Ceiling function: quasiconvex but not convex"""
    return np.ceil(x)


# Plot 1D examples
x = np.linspace(-3, 3, 1000)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Example 1
axs[0].plot(x, example1(x))
axs[0].set_title("Example 1: |x|")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].grid(True)

# Example 2
axs[1].plot(x, example2(x))
axs[1].set_title("Example 2: 1/(1+x²)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].grid(True)

# Example 5
axs[2].plot(x, example5(x))
axs[2].set_title("Example 5: Ceiling Function")
axs[2].set_xlabel("x")
axs[2].set_ylabel("f(x)")
axs[2].grid(True)

plt.tight_layout()
plt.savefig("quasiconvex_1d_examples.png")

# Plot 2D examples
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Example 3: 3D plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
Z1 = example3(X, Y)
surf1 = ax1.plot_surface(X, Y, Z1, cmap="viridis", alpha=0.8)
ax1.set_title("Example 3: max(|x|, |y|)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")

# Example 4: 3D plot
ax2 = fig.add_subplot(122, projection="3d")
Z2 = example4(X, Y)
surf2 = ax2.plot_surface(X, Y, Z2, cmap="plasma", alpha=0.8)
ax2.set_title("Example 4: (x²+y²)/(1+x²+y²)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x,y)")

plt.tight_layout()
plt.savefig("quasiconvex_2d_examples.png")


# Demonstrate quasiconvexity property
def demonstrate_quasiconvexity():
    """Demonstrate the quasiconvexity property for example 2"""
    x = np.linspace(-3, 3, 1000)
    f_x = example2(x)

    # Choose two points
    x1, x2 = -2.0, 2.0
    f_x1, f_x2 = example2(x1), example2(x2)

    # Calculate the line segment points
    lambdas = np.linspace(0, 1, 100)
    x_lambdas = [x1 * lambda_val + x2 * (1 - lambda_val) for lambda_val in lambdas]
    f_x_lambdas = [example2(x_lambda) for x_lambda in x_lambdas]
    max_f = max(f_x1, f_x2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, f_x, "b-", label="f(x) = 1/(1+x²)")
    plt.plot(x_lambdas, f_x_lambdas, "r--", linewidth=2, label="f(λx₁ + (1-λ)x₂)")
    plt.axhline(y=max_f, color="g", linestyle="-.", label="max{f(x₁), f(x₂)}")
    plt.scatter([x1, x2], [f_x1, f_x2], color="k", s=100, label="Points x₁, x₂")

    plt.title("Demonstration of Quasiconvexity")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("quasiconvexity_demo.png")

    # Print some insights
    print(f"Function value at x1={x1}: {f_x1:.4f}")
    print(f"Function value at x2={x2}: {f_x2:.4f}")
    print(f"Maximum of f(x1) and f(x2): {max_f:.4f}")
    print(
        f"All points on the line segment have f(x) ≤ {max_f:.4f}, demonstrating quasiconvexity"
    )


# Example of verifying quasiconvexity using the definition
def verify_quasiconvexity(func, x1, x2, num_points=100):
    """Verify if a function satisfies the quasiconvexity property between two points"""
    lambdas = np.linspace(0, 1, num_points)
    max_value = max(func(x1), func(x2))

    for lambda_val in lambdas:
        x_lambda = lambda_val * x1 + (1 - lambda_val) * x2
        f_x_lambda = func(x_lambda)
        if (
            f_x_lambda > max_value + 1e-10
        ):  # Add small tolerance for numerical stability
            return False

    return True


# Run the verification on our examples
if __name__ == "__main__":
    # Demonstrate quasiconvexity visually
    demonstrate_quasiconvexity()

    # Verify quasiconvexity for some example points
    test_points = [(-2, 2), (-1, 3), (0, 1), (1, 4)]

    print("\nVerifying quasiconvexity for Example 1: |x|")
    for x1, x2 in test_points:
        result = verify_quasiconvexity(example1, x1, x2)
        print(
            f"Between points {x1} and {x2}: {'Quasiconvex ✓' if result else 'Not quasiconvex ✗'}"
        )

    print("\nVerifying quasiconvexity for Example 2: 1/(1+x²)")
    for x1, x2 in test_points:
        result = verify_quasiconvexity(example2, x1, x2)
        print(
            f"Between points {x1} and {x2}: {'Quasiconvex ✓' if result else 'Not quasiconvex ✗'}"
        )

    print("\nVerifying quasiconvexity for Example 5: Ceiling function")
    for x1, x2 in test_points:
        result = verify_quasiconvexity(example5, x1, x2)
        print(
            f"Between points {x1} and {x2}: {'Quasiconvex ✓' if result else 'Not quasiconvex ✗'}"
        )
