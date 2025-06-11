import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Define the polynomial coefficients (for a cubic polynomial)
coefficients = [1, 0, 0, 0, 0]  # This represents x^3 - 2x^2 + 1

# Create x values from -1 to 1
x = np.linspace(-1, 1, 100)

# Evaluate the polynomial
y = np.polyval(coefficients, x)

# Initial slope and intercept
initial_slope = 1.25
initial_intercept = 2  # This ensures difference at x=-2 is 15

# Create the figure and the line that we will manipulate
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
plt.subplots_adjust(bottom=0.3)  # Increased bottom margin to accommodate sliders

# Plot the cubic polynomial
(cubic_line,) = ax1.plot(x, y, "b-", linewidth=2, label="f(x) = x^4")

# Initial linear function
y_linear = initial_slope * x + initial_intercept
(linear_line,) = ax1.plot(
    x,
    y_linear,
    "r--",
    linewidth=2,
    label=f"Linear: {initial_slope:.2f}x + {initial_intercept:.2f}",
)

# Initial filled area
fill1 = ax1.fill_between(x, y_linear, y, where=(y_linear > y), color="green", alpha=0.3)
fill2 = ax1.fill_between(x, y_linear, y, where=(y_linear < y), color="red", alpha=0.3)

# Initial max difference point
difference = y_linear - y  # Calculate actual difference (linear - polynomial)
max_diff_idx = np.argmax(difference)  # Find maximum of the actual difference
max_diff_x = x[max_diff_idx]
max_diff_y1 = y[max_diff_idx]
max_diff_y2 = y_linear[max_diff_idx]
(max_point1,) = ax1.plot(max_diff_x, max_diff_y1, "o", color="purple", markersize=8)
(max_point2,) = ax1.plot(max_diff_x, max_diff_y2, "o", color="purple", markersize=8)

# Calculate and plot conjugate function
slopes = np.linspace(-5, 5, 100)
conjugate_values = []
for s in slopes:
    # For each slope, find the maximum difference
    differences = s * x - y
    conjugate_values.append(np.max(differences))

# Plot conjugate function
(conjugate_line,) = ax2.plot(
    slopes, conjugate_values, "g-", linewidth=2, label="Conjugate f*(y)"
)
(current_point,) = ax2.plot([initial_slope], [np.max(difference)], "ro", markersize=8)

# Add text annotation for coordinates
coord_text = ax2.text(
    0.02,
    0.98,
    f"Current point: (y={initial_slope:.2f}, f*(y)={np.max(difference):.2f})",
    transform=ax2.transAxes,
    verticalalignment="top",
)

ax1.set_title("Interactive Plot: Adjust the sliders to modify the linear function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)
ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
ax1.legend()

ax2.set_title("Conjugate Function f*(y)")
ax2.set_xlabel("Slope (y)")
ax2.set_ylabel("f*(y)")
ax2.grid(True)
ax2.legend()

# Create sliders
ax_slope = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_intercept = plt.axes([0.25, 0.1, 0.65, 0.03])

slope_slider = Slider(
    ax=ax_slope,
    label="Slope",
    valmin=-5,
    valmax=5,
    valinit=initial_slope,
)

intercept_slider = Slider(
    ax=ax_intercept,
    label="Intercept",
    valmin=-20,
    valmax=20,
    valinit=initial_intercept,
)


# Update function for the sliders
def update(val):
    slope = slope_slider.val
    intercept = intercept_slider.val
    y_linear = slope * x + intercept

    # Update linear line
    linear_line.set_ydata(y_linear)
    linear_line.set_label(f"Linear: {slope:.2f}x + {intercept:.2f}")

    # Update filled areas
    global fill1, fill2
    fill1.remove()
    fill2.remove()
    fill1 = ax1.fill_between(
        x, y_linear, y, where=(y_linear > y), color="green", alpha=0.3
    )
    fill2 = ax1.fill_between(
        x, y_linear, y, where=(y_linear < y), color="red", alpha=0.3
    )

    # Update max difference points
    difference = y_linear - y  # Calculate actual difference (linear - polynomial)
    max_diff_idx = np.argmax(difference)  # Find maximum of the actual difference
    max_diff_x = x[max_diff_idx]
    max_diff_y1 = y[max_diff_idx]
    max_diff_y2 = y_linear[max_diff_idx]
    max_point1.set_data([max_diff_x], [max_diff_y1])
    max_point2.set_data([max_diff_x], [max_diff_y2])

    # Update conjugate plot point
    current_point.set_data([slope], [np.max(difference)])

    # Update coordinate text
    coord_text.set_text(
        f"Current point: (y={slope:.2f}, f*(y)={np.max(difference):.2f})"
    )

    # Update legend
    ax1.legend()
    fig.canvas.draw_idle()


# Register the update function with both sliders
slope_slider.on_changed(update)
intercept_slider.on_changed(update)

plt.show()
