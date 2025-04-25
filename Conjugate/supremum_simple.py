
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Example 1: Finite set with a maximum
S1 = [1, 3, 4, 7, 9]
supremum1 = max(S1)

# Plot the set points
ax1.scatter(S1, [0]*len(S1), s=100, color='blue', label='Set elements')
ax1.axhline(y=0, color='black', linewidth=0.5)  # Number line
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlim(0, 11)

# Mark the supremum
ax1.scatter([supremum1], [0], s=150, color='red', marker='s', label='Supremum')
ax1.annotate(f"sup(S) = {supremum1}", xy=(supremum1, 0), xytext=(supremum1, 0.2),
            arrowprops=dict(arrowstyle='->'), color='red', ha='center')

# Set title and labels
ax1.set_title("Set with Maximum: sup(S) = max(S)")
ax1.set_xlabel("Real Number Line")
ax1.set_yticks([])

# Example 2: Set without maximum (sequence converging to 1)
S2 = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875]
supremum2 = 1  # The sequence approaches 1 but never reaches it

# Plot the set points
ax2.scatter(S2, [0]*len(S2), s=100, color='blue', label='Set elements')
ax2.axhline(y=0, color='black', linewidth=0.5)  # Number line
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlim(-0.1, 1.2)

# Mark the supremum
ax2.scatter([supremum2], [0], s=150, color='red', marker='s', label='Supremum')
ax2.annotate(f"sup(S) = {supremum2}", xy=(supremum2, 0), xytext=(supremum2, 0.2),
            arrowprops=dict(arrowstyle='->'), color='red', ha='center')

# Set title and labels
ax2.set_title("Set without Maximum: sup(S) ∉ S")
ax2.set_xlabel("Real Number Line")
ax2.set_yticks([])

# Add an overall title
plt.suptitle("The Concept of Supremum", fontsize=16)
plt.tight_layout()

# Save the figure
plt.savefig('/Users/carsten/Documents/Science/supremum_simple.png', dpi=300, bbox_inches='tight')

print("Visualizations saved to /Users/carsten/Documents/Science/supremum_simple.png")
