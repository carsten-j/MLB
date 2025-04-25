import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def example_function(x):
    """Our example function f(x) = x^2"""
    return x**2

def compute_conjugate(f, lambda_val, x_range=(-10, 10)):
    """Compute the conjugate function value f*(λ)"""
    # Define the objective function to maximize: λx - f(x)
    def objective(x):
        return lambda_val * x - f(x)
    
    # Find the x that maximizes λx - f(x)
    neg_objective = lambda x: -objective(x)
    result = minimize_scalar(neg_objective, bounds=x_range, method='bounded')
    optimal_x = result.x
    conj_val = objective(optimal_x)
    
    return conj_val, optimal_x

def visualize_vertical_distance(f, lambda_val, x_range=(-3, 3)):
    """Visualize the vertical distance interpretation of conjugate functions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate points for plotting the function
    x_vals = np.linspace(x_range[0], x_range[1], 1000)
    y_vals = [f(x) for x in x_vals]
    
    # Plot the original function
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    
    # Compute the conjugate value and optimal x
    conj_val, optimal_x = compute_conjugate(f, lambda_val)
    optimal_y = f(optimal_x)
    
    # Plot the initial line λx (before shifting)
    initial_line_vals = [lambda_val * x for x in x_vals]
    ax.plot(x_vals, initial_line_vals, 'r--', alpha=0.5, label=f'Initial line y = {lambda_val}x')
    
    # Plot the final supporting line λx - f*(λ)
    final_line_vals = [lambda_val * x - conj_val for x in x_vals]
    ax.plot(x_vals, final_line_vals, 'r-', linewidth=2, label=f'Supporting line y = {lambda_val}x - {conj_val:.2f}')
    
    # Mark the optimal point
    ax.plot([optimal_x], [optimal_y], 'ro', markersize=8)
    ax.annotate(f'Contact point: ({optimal_x:.2f}, {optimal_y:.2f})', 
                xy=(optimal_x, optimal_y), 
                xytext=(optimal_x + 0.5, optimal_y + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Show the vertical distance between initial line and final line
    ax.vlines(x=0, ymin=-conj_val, ymax=0, color='g', linestyle='-', linewidth=2)
    ax.annotate(f'f*({lambda_val}) = {conj_val:.2f}', 
                xy=(0, -conj_val/2), 
                xytext=(0.5, -conj_val/2),
                arrowprops=dict(facecolor='green', shrink=0.05))
    
    # Visual aids: Show λx - f(x) at the optimal point
    ax.vlines(x=optimal_x, ymin=optimal_y, ymax=lambda_val*optimal_x, color='m', linestyle='--', linewidth=1.5)
    ax.annotate(f'λx - f(x) = {conj_val:.2f}', 
                xy=(optimal_x, (optimal_y + lambda_val*optimal_x)/2), 
                xytext=(optimal_x + 0.5, (optimal_y + lambda_val*optimal_x)/2),
                arrowprops=dict(facecolor='magenta', shrink=0.05))
    
    # Plot styling
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Visualizing Conjugate Function for λ = {lambda_val}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'vertical_distance_lambda_{lambda_val}.png', dpi=300)
    plt.close()

def create_multiple_visualizations():
    """Create visualizations for multiple λ values"""
    lambda_vals = [-2, -1, 0, 1, 2]
    
    for lam in lambda_vals:
        visualize_vertical_distance(example_function, lam)
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    create_multiple_visualizations()
