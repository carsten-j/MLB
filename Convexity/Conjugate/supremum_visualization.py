
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Set up figure and axis
plt.figure(figsize=(12, 6))

# Example 1: Bounded set with maximum element
def plot_bounded_with_max():
    ax = plt.subplot(121)
    
    # Define our set: [0, 1, 2, 3, 4, 5]
    S = np.array([0, 1, 2, 3, 4, 5])
    
    # Plot the set elements
    ax.scatter(S, np.zeros_like(S), s=100, color='blue', zorder=3)
    
    # Plot the real number line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-0.5, 0.5)
    
    # Mark the supremum with a red square
    supremum = 5
    ax.scatter([supremum], [0], s=150, color='red', marker='s', zorder=4)
    
    # Add annotations
    ax.annotate("Elements of set S", xy=(2.5, -0.15), xytext=(2.5, -0.3),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.annotate(f"sup(S) = {supremum}\n(also max(S))", xy=(supremum, 0), xytext=(supremum, 0.3),
                arrowprops=dict(arrowstyle='->'), ha='center', color='red')
    
    ax.set_title("Set with Maximum Element")
    ax.set_xlabel("Real Number Line")
    ax.set_yticks([])
    
# Example 2: Bounded set without maximum element
def plot_bounded_without_max():
    ax = plt.subplot(122)
    
    # Define our set: [0, 0.5, 0.75, 0.875, 0.9375, ...] approaching 1
    x = np.linspace(0, 0.99, 1000)
    S = np.array([0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875])
    
    # Plot the set elements
    ax.scatter(S, np.zeros_like(S), s=100, color='blue', zorder=3)
    
    # Plot the real number line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 0.5)
    
    # Mark the supremum with a red square
    supremum = 1
    ax.scatter([supremum], [0], s=150, color='red', marker='s', zorder=4)
    
    # Add annotations
    ax.annotate("Elements of set S", xy=(0.5, -0.15), xytext=(0.5, -0.3),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax.annotate(f"sup(S) = {supremum}\n(not in S)", xy=(supremum, 0), xytext=(supremum, 0.3),
                arrowprops=dict(arrowstyle='->'), ha='center', color='red')
    
    ax.set_title("Set without Maximum Element")
    ax.set_xlabel("Real Number Line")
    ax.set_yticks([])

# Create the plots
plot_bounded_with_max()
plot_bounded_without_max()

# Add a legend and overall title
plt.suptitle("Visualization of Supremum", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Add text explaining the concept
plt.figtext(0.5, 0.01, 
            "The supremum (sup) is the least upper bound of a set.\n"
            "Left: For a set with maximum, sup(S) = max(S)\n"
            "Right: For a set without maximum (e.g., sequence approaching 1), sup(S) = 1 even though 1 is not in the set",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.savefig('/Users/carsten/Documents/Science/supremum_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional visualization: Comparing different types of sets
plt.figure(figsize=(10, 8))

# Example sets with their suprema
sets = [
    {"name": "Finite set", "elements": [1, 3, 5, 7, 9], "supremum": 9},
    {"name": "Bounded infinite set", "elements": np.linspace(0, 0.99, 20), "supremum": 1},
    {"name": "Open interval (0,1)", "elements": np.linspace(0.05, 0.95, 20), "supremum": 1},
    {"name": "Closed interval [0,1]", "elements": np.linspace(0, 1, 20), "supremum": 1},
]

for i, s in enumerate(sets):
    ax = plt.subplot(4, 1, i+1)
    
    # Plot the set elements
    ax.scatter(s["elements"], np.zeros_like(s["elements"]), s=80, color='blue', zorder=3)
    
    # Plot the real number line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set appropriate limits
    if i == 0:  # Finite set
        ax.set_xlim(0, 10)
    else:
        ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 0.3)
    
    # Mark the supremum
    ax.scatter([s["supremum"]], [0], s=120, color='red', marker='s', zorder=4)
    
    # Add annotations
    if i == 0:  # Finite set
        ax.annotate(f"sup(S) = {s['supremum']}", xy=(s["supremum"], 0), 
                    xytext=(s["supremum"]+0.5, 0.15), ha='center', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    else:
        ax.annotate(f"sup(S) = {s['supremum']}", xy=(s["supremum"], 0), 
                    xytext=(s["supremum"]-0.3, 0.15), ha='center', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_title(s["name"])
    ax.set_yticks([])
    
    # Add a special marker for the maximum element if it exists in the set
    if i == 0 or i == 3:  # Finite set or closed interval
        max_element = max(s["elements"])
        ax.scatter([max_element], [0], s=200, facecolors='none', edgecolors='green', 
                  linewidth=2, zorder=5)
        ax.annotate("max(S)", xy=(max_element, 0), xytext=(max_element, -0.15), 
                    ha='center', color='green', arrowprops=dict(arrowstyle='->', color='green'))

plt.suptitle("Supremum Across Different Types of Sets", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/carsten/Documents/Science/supremum_different_sets.png', dpi=300, bbox_inches='tight')
plt.show()

# Interactive visualization of convergence to supremum
plt.figure(figsize=(10, 6))

# Define a sequence: 1 - 1/n which approaches 1 as n increases
n_values = np.arange(1, 101)
sequence = 1 - 1/n_values

plt.plot(n_values, sequence, 'bo-', alpha=0.5)
plt.axhline(y=1, color='r', linestyle='--', label='Supremum = 1')
plt.xlabel('n')
plt.ylabel('1 - 1/n')
plt.title('Sequence Approaching its Supremum')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/carsten/Documents/Science/supremum_convergence.png', dpi=300)
plt.show()
