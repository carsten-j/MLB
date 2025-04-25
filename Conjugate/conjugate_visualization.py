import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def quadratic(x, a=1):
    """Simple quadratic function f(x) = (a/2)x^2"""
    return (a/2) * x**2

def quadratic_conjugate(lambda_val, a=1):
    """Conjugate of quadratic function: f*(λ) = λ^2/(2a)"""
    return lambda_val**2 / (2*a)

def exponential(x):
    """Exponential function f(x) = e^x"""
    return np.exp(x)

def exponential_conjugate(lambda_val):
    """Conjugate of exponential function: f*(λ) = λ*log(λ) - λ, for λ > 0"""
    if lambda_val <= 0:
        return float('inf')  # Domain constraint
    return lambda_val * np.log(lambda_val) - lambda_val

def compute_conjugate_numerically(f, lambda_val, x_range=(-10, 10)):
    """Compute conjugate value by numerical optimization"""
    # For a given λ, find x that maximizes λx - f(x)
    neg_objective = lambda x: -(lambda_val * x - f(x))
    result = minimize_scalar(neg_objective, bounds=x_range, method='bounded')
    optimal_x = result.x
    return lambda_val * optimal_x - f(optimal_x), optimal_x

def plot_function_and_tangent(f, x_range, lambda_val, ax, a=1, title=""):
    """Plot a function and its tangent line with slope λ"""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(xi, a) if callable(f) and f.__name__ == 'quadratic' else f(xi) for xi in x]
    
    # Compute conjugate value and the optimal x
    if f.__name__ == 'quadratic':
        conj_value = quadratic_conjugate(lambda_val, a)
        optimal_x = lambda_val / a  # For quadratic, x = λ/a where f'(x) = λ
    elif f.__name__ == 'exponential':
        conj_value = exponential_conjugate(lambda_val)
        optimal_x = np.log(lambda_val) if lambda_val > 0 else 0  # For exp, x = log(λ) where f'(x) = λ
    else:
        conj_value, optimal_x = compute_conjugate_numerically(f, lambda_val, x_range)
    
    # Plot the function
    ax.plot(x, y, 'b-', label=f.__name__)
    
    # Plot the tangent/supporting line
    tangent_y = lambda_val * x - conj_value
    ax.plot(x, tangent_y, 'r--', label=f'Tangent with slope λ={lambda_val}')
    
    # Mark the point where the tangent touches the function
    if f.__name__ == 'quadratic':
        touch_y = f(optimal_x, a)
    else:
        touch_y = f(optimal_x)
    ax.plot([optimal_x], [touch_y], 'ro', markersize=8)
    
    # Add dashed lines to show the intercept
    ax.plot([0], [-conj_value], 'go', markersize=8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add text annotations
    ax.annotate(f'f*({lambda_val}) = {conj_value:.2f}', 
                xy=(0, -conj_value), 
                xytext=(0.5, -conj_value - 0.5), 
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10)
    
    ax.annotate(f'Tangent point: ({optimal_x:.2f}, {touch_y:.2f})', 
                xy=(optimal_x, touch_y), 
                xytext=(optimal_x + 1, touch_y + 1), 
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Create multiple visualizations with different slopes
def create_slope_visualizations():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Quadratic function with different slopes
    plot_function_and_tangent(quadratic, (-5, 5), lambda_val=1, ax=axes[0, 0], 
                              title="Quadratic Function f(x) = x²/2 with λ=1")
    plot_function_and_tangent(quadratic, (-5, 5), lambda_val=2, ax=axes[0, 1], 
                              title="Quadratic Function f(x) = x²/2 with λ=2")
    
    # Exponential function with different slopes
    plot_function_and_tangent(exponential, (-2, 4), lambda_val=1, ax=axes[1, 0], 
                              title="Exponential Function f(x) = e^x with λ=1")
    plot_function_and_tangent(exponential, (-2, 4), lambda_val=2, ax=axes[1, 1], 
                              title="Exponential Function f(x) = e^x with λ=2")
    
    plt.tight_layout()
    plt.savefig("slope_representation.png", dpi=300)
    plt.close()

# Plot the conjugate functions themselves
def plot_conjugate_functions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot quadratic and its conjugate
    lambda_vals = np.linspace(0, 3, 100)
    conjugate_vals = [quadratic_conjugate(lam) for lam in lambda_vals]
    
    ax1.plot(lambda_vals, conjugate_vals, 'g-', linewidth=2)
    ax1.set_title("Conjugate of f(x) = x²/2")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("f*(λ)")
    ax1.grid(True, alpha=0.3)
    
    # Plot exponential and its conjugate
    lambda_vals = np.linspace(0.1, 3, 100)
    conjugate_vals = [exponential_conjugate(lam) for lam in lambda_vals]
    
    ax2.plot(lambda_vals, conjugate_vals, 'g-', linewidth=2)
    ax2.set_title("Conjugate of f(x) = e^x")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("f*(λ)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("conjugate_functions.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    create_slope_visualizations()
    plot_conjugate_functions()
    print("Visualizations created successfully!")
