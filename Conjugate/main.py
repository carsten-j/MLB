import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar


# Define our function f(x) = x²(x-1)(x+1)
def f(x):
    return x**2 * (x - 1) * (x + 1)


# Calculate the derivative of f(x) = x²(x-1)(x+1)
def f_prime(x):
    # Expanded: f(x) = x⁴ - x²
    # f'(x) = 4x³ - 2x
    return 4 * x**3 - 2 * x


# Function to calculate conjugate value for a specific y
def calc_conjugate(y):
    # We want to maximize yx - f(x)
    # This happens where y = f'(x)
    result = minimize_scalar(lambda x: -(y * x - f(x)))
    x_max = result.x
    return x_max, y * x_max - f(x_max)


# Set up the figure with multiple subplots
fig = plt.figure(figsize=(12, 15))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0])  # Original function
ax2 = fig.add_subplot(gs[1])  # Conjugate function
ax3 = fig.add_subplot(gs[2])  # Specific case for x = 0.75

# Define the x range and compute function values
x_vals = np.linspace(-2, 2, 1000)
f_vals = [f(x) for x in x_vals]

# Plot the original function
ax1.plot(x_vals, f_vals, "b-", linewidth=2, label="f(x) = x²(x-1)(x+1)")
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("f(x)", fontsize=12)
ax1.set_title("Original Function f(x)", fontsize=14)
ax1.grid(True)
ax1.legend(fontsize=10)

# Compute and plot the convex conjugate function
# We need to determine the range of y values
# Since y = f'(x), we can compute f'(x) for our x range
y_vals = [f_prime(x) for x in x_vals]
y_vals = sorted(set([round(y, 2) for y in y_vals[::20]]))  # Take sample of y values

# Calculate conjugate function values for each y
conjugate_points = [calc_conjugate(y) for y in y_vals]
x_maxs, f_star_vals = zip(*conjugate_points)

# Plot the conjugate function
ax2.plot(y_vals, f_star_vals, "g-", linewidth=2, label="f*(y)")
ax2.set_xlabel("y", fontsize=12)
ax2.set_ylabel("f*(y)", fontsize=12)
ax2.set_title("Convex Conjugate Function f*(y)", fontsize=14)
ax2.grid(True)
ax2.legend(fontsize=10)

# Special highlight for our specific example at x = 0.75
x_specific = 0.75
y_specific = f_prime(x_specific)  # This is the slope of the tangent line at x = 0.75
f_x_specific = f(x_specific)
conjugate_value = x_specific * y_specific - f_x_specific

# Add this specific point to both plots
ax1.plot(x_specific, f_x_specific, "ro", markersize=8)
ax1.annotate(
    f"x = {x_specific}\nf(x) = {f_x_specific:.4f}",
    xy=(x_specific, f_x_specific),
    xytext=(x_specific + 0.3, f_x_specific + 0.3),
    fontsize=10,
    arrowprops=dict(facecolor="red", shrink=0.05),
)

# Plot the tangent line at x = 0.75
tangent_x = np.linspace(-2, 2, 100)
tangent_y = y_specific * (tangent_x - x_specific) + f_x_specific
ax1.plot(
    tangent_x,
    tangent_y,
    "r--",
    linewidth=1.5,
    label=f"Tangent at x = {x_specific} with slope y = {y_specific:.4f}",
)

# Mark the corresponding point on f*(y)
ax2.plot(y_specific, conjugate_value, "ro", markersize=8)
ax2.annotate(
    f"y = {y_specific:.4f}\nf*(y) = {conjugate_value:.4f}",
    xy=(y_specific, conjugate_value),
    xytext=(y_specific + 0.3, conjugate_value + 0.3),
    fontsize=10,
    arrowprops=dict(facecolor="red", shrink=0.05),
)

# Third subplot: Visualization of yx - f(x) to find the maximum
# This corresponds to finding f*(y) for our specific y value
x_range = np.linspace(-2, 2, 500)
objective_vals = [y_specific * x - f(x) for x in x_range]

ax3.plot(x_range, objective_vals, "b-", linewidth=2, label=f"{y_specific:.4f}x - f(x)")
ax3.axhline(
    y=conjugate_value,
    color="r",
    linestyle="--",
    label=f"Max value = f*({y_specific:.4f}) = {conjugate_value:.4f}",
)
ax3.axvline(x=x_specific, color="g", linestyle="--", label=f"Max at x = {x_specific}")
ax3.plot(x_specific, conjugate_value, "ro", markersize=8)
ax3.set_xlabel("x", fontsize=12)
ax3.set_ylabel(f"{y_specific:.4f}x - f(x)", fontsize=12)
ax3.set_title(
    f"Finding f*({y_specific:.4f}) as max[ {y_specific:.4f}x - f(x) ]", fontsize=14
)
ax3.grid(True)
ax3.legend(fontsize=10)

# Add calculation steps in text
calculation_steps = (
    f"Calculation of f*(y) when x = 0.75:\n\n"
    f"1. Calculate the derivative: f'(0.75) = {f_prime(0.75):.4f}\n"
    f"2. This gives us y = {y_specific:.4f}\n"
    f"3. Calculate f(0.75) = {f_x_specific:.4f}\n"
    f"4. Compute f*(y) = xy - f(x) = 0.75 × {y_specific:.4f} - {f_x_specific:.4f}\n"
    f"5. f*({y_specific:.4f}) = {conjugate_value:.4f}"
)
fig.text(
    0.15,
    0.02,
    calculation_steps,
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
)

ax1.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.15)
plt.show()
