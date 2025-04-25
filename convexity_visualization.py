import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define our functions
def convex_but_not_strongly(x):
    return x**4  # f''(x) = 12x² which equals 0 at x=0

def strongly_convex(x):
    return x**4 + 0.5*x**2  # f''(x) = 12x² + 1 which is always ≥ 1

# Sample points
x = np.linspace(-2, 2, 1000)

# Create plot
plt.figure(figsize=(12, 8))

# Plot the functions
plt.subplot(2, 1, 1)
plt.plot(x, convex_but_not_strongly(x), 'g-', label='Convex: f(x) = x⁴')
plt.plot(x, strongly_convex(x), 'b-', label='Strongly convex: f(x) = x⁴ + 0.5x²')

# Highlight near x=0 with an inset
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True)
plt.legend()
plt.title('Convex vs Strongly Convex Functions')
plt.ylabel('f(x)')

# Plot the second derivatives
plt.subplot(2, 1, 2)
plt.plot(x, 12*x**2, 'g-', label='f\'\'(x) for x⁴ = 12x²')
plt.plot(x, 12*x**2 + 1, 'b-', label='f\'\'(x) for x⁴ + 0.5x² = 12x² + 1')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='b', linestyle='--', alpha=0.5, label='μ = 1 (minimum curvature)')
plt.grid(True)
plt.legend()
plt.title('Second Derivatives (Curvature)')
plt.xlabel('x')
plt.ylabel('f\'\'(x)')
plt.tight_layout()

# Generate a separate zoomed-in plot focusing on x=0
plt.figure(figsize=(12, 6))

# Create closely spaced points near x=0 for detailed view
x_zoom = np.linspace(-0.5, 0.5, 1000)

# Plot the functions
plt.subplot(1, 2, 1)
plt.plot(x_zoom, convex_but_not_strongly(x_zoom), 'g-', label='Convex: f(x) = x⁴')
plt.plot(x_zoom, strongly_convex(x_zoom), 'b-', label='Strongly convex: f(x) = x⁴ + 0.5x²')
plt.grid(True)
plt.legend()
plt.title('Zoomed View Near x=0')
plt.xlabel('x')
plt.ylabel('f(x)')

# Illustrate the "bending" with chords
plt.subplot(1, 2, 2)

# Generate points on the x^4 curve
x_points = [-0.4, 0.4]
y_convex = [convex_but_not_strongly(x) for x in x_points]
y_strongly = [strongly_convex(x) for x in x_points]

# Draw function curves
x_fine = np.linspace(x_points[0], x_points[1], 100)
plt.plot(x_fine, convex_but_not_strongly(x_fine), 'g-', label='Convex: f(x) = x⁴')
plt.plot(x_fine, strongly_convex(x_fine), 'b-', label='Strongly convex: f(x) = x⁴ + 0.5x²')

# Draw chords
plt.plot(x_points, y_convex, 'g--', label='Chord for x⁴')
plt.plot(x_points, y_strongly, 'b--', label='Chord for x⁴ + 0.5x²')

# Shade the areas between chord and function
# For convex function
chord_y_convex = np.interp(x_fine, x_points, y_convex)
func_y_convex = convex_but_not_strongly(x_fine)
vertices_convex = list(zip(x_fine, chord_y_convex)) + list(zip(x_fine[::-1], func_y_convex[::-1]))
poly_convex = Polygon(vertices_convex, facecolor='g', alpha=0.2)
plt.gca().add_patch(poly_convex)

# For strongly convex function
chord_y_strongly = np.interp(x_fine, x_points, y_strongly)
func_y_strongly = strongly_convex(x_fine)
vertices_strongly = list(zip(x_fine, chord_y_strongly)) + list(zip(x_fine[::-1], func_y_strongly[::-1]))
poly_strongly = Polygon(vertices_strongly, facecolor='b', alpha=0.2)
plt.gca().add_patch(poly_strongly)

plt.grid(True)
plt.legend()
plt.title('Distance Between Function and Chord')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.tight_layout()
plt.savefig('convexity_comparison.png')
plt.show()

# Print mathematical explanation
print("\nMathematical explanation of strong convexity:")
print("For a strongly convex function with parameter μ > 0:")
print("f(tx + (1-t)y) ≤ tf(x) + (1-t)f(y) - μt(1-t)||x-y||²")
print("\nIn our example:")
print("- f(x) = x⁴ is convex but not strongly convex (its second derivative equals 0 at x=0)")
print("- g(x) = x⁴ + 0.5x² is strongly convex (its second derivative is always ≥ 1)")
