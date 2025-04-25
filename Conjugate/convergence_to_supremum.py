
import numpy as np
import matplotlib.pyplot as plt

# Create a figure to demonstrate convergence to supremum
plt.figure(figsize=(10, 6))

# Example: Sequence a_n = 1 - 1/n approaches 1 as n increases
n_values = np.arange(1, 51)
sequence = 1 - 1/n_values

# Plot the sequence values
plt.plot(n_values, sequence, 'bo-', markersize=5, alpha=0.7, label='Sequence a_n = 1 - 1/n')

# Plot the supremum as a horizontal line
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Supremum = 1')

# Highlight specific points
highlight_indices = [1, 2, 4, 10, 25, 50]
highlight_values = [1 - 1/n for n in highlight_indices]
plt.scatter(highlight_indices, highlight_values, color='green', s=100, zorder=3)

# Annotate the highlighted points
for i, n in enumerate(highlight_indices):
    val = 1 - 1/n
    plt.annotate(f'n={n}, a_n={val:.4f}', 
                 xy=(n, val), 
                 xytext=(n+1, val-0.05 if i % 2 == 0 else val+0.05),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9)

# Customize the plot
plt.title('Convergence to Supremum: Sequence a_n = 1 - 1/n', fontsize=14)
plt.xlabel('n (term number)', fontsize=12)
plt.ylabel('Value of a_n', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Add explanatory text
plt.figtext(0.5, 0.01, 
            "The sequence gets arbitrarily close to its supremum (1), but never reaches it.\n"
            "This demonstrates why supremum differs from maximum in infinite sets.",
            ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

# Set axis limits for better visualization
plt.xlim(0, 55)
plt.ylim(0, 1.1)

# Save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('/Users/carsten/Documents/Science/convergence_to_supremum.png', dpi=300, bbox_inches='tight')

print("Convergence visualization saved to /Users/carsten/Documents/Science/convergence_to_supremum.png")
