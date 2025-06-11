import matplotlib.pyplot as plt
import numpy as np


# A correct strongly convex function
def strongly_convex(x):
    return 2 * x**2  # Second derivative is always 4


# A standard convex function
def convex(x):
    return x**2  # Second derivative is always 2


# A linear function
def linear(x):
    return 2 * x  # Second derivative is always 0


# Our problematic function that's not strongly convex everywhere
def not_strongly_convex(x):
    return x**2 + 2 * np.sin(x) ** 2


x = np.linspace(-3, 3, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x, linear(x), "r-", label="Linear: f(x) = 2x")
plt.plot(x, convex(x), "g-", label="Convex: f(x) = x²")
plt.plot(x, strongly_convex(x), "b-", label="Strongly convex: f(x) = 2x²")
plt.plot(
    x,
    not_strongly_convex(x),
    "m-",
    label="Not strongly convex everywhere: x² + 2sin²(x)",
)

plt.grid(True)
plt.legend()
plt.title("Comparison of Function Types")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
