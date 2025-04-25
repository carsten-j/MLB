import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown, fixed
import ipywidgets as widgets
from IPython.display import display, Markdown, Math

"""
An interactive tool to visualize Lagrangian duality for simple optimization problems.
This demonstrates how the dual function is derived and how strong duality works.
"""

def plot_lagrangian_duality(problem_type='quadratic', lambda_val=1.0):
    """
    Visualize the Lagrangian approach for different optimization problems.
    
    Parameters:
    -----------
    problem_type : str
        Type of problem ('quadratic', 'exponential', or 'entropy')
    lambda_val : float
        Value of the Lagrange multiplier
    """
    plt.figure(figsize=(15, 10))
    
    # Define the domain
    x = np.linspace(0.01, 5, 1000)
    
    # Define objective functions and constraints based on problem type
    if problem_type == 'quadratic':
        f = lambda x: x**2
        g = lambda x: x - 2  # constraint: x <= 2
        title = "Minimize $f(x) = x^2$ subject to $x \\leq 2$"
        optimal_x = 0  # Unconstrained optimum
        constrained_x = 0  # Constrained optimum (satisfies the constraint)
    
    elif problem_type == 'exponential':
        f = lambda x: np.exp(x)
        g = lambda x: x - 1  # constraint: x <= 1
        title = "Minimize $f(x) = e^x$ subject to $x \\leq 1$"
        optimal_x = -np.inf  # Unconstrained optimum would be -∞
        constrained_x = 1  # Constrained optimum
    
    elif problem_type == 'entropy':
        f = lambda x: x * np.log(x)
        g = lambda x: 1 - x  # constraint: x >= 1
        title = "Minimize $f(x) = x\\log(x)$ subject to $x \\geq 1$"
        optimal_x = 1/np.e  # Unconstrained optimum
        constrained_x = 1  # Constrained optimum
    
    # Lagrangian function L(x, λ) = f(x) + λ*g(x)
    lagrangian = lambda x, lam: f(x) + lam * g(x)
    
    # Calculate L(x, λ) for the given λ
    L_x = lagrangian(x, lambda_val)
    
    # Find x that minimizes L(x, λ) for the given λ
    if problem_type == 'quadratic':
        x_min_L = 0 if lambda_val <= 0 else min(0, 2 - lambda_val/2)
    elif problem_type == 'exponential':
        x_min_L = -np.inf if lambda_val <= 0 else np.log(lambda_val) if lambda_val > 0 else 1
        x_min_L = min(x_min_L, 1)  # Cap at the constraint boundary
    elif problem_type == 'entropy':
        x_min_L = np.exp(-lambda_val)
        x_min_L = max(x_min_L, 1)  # Cap at the constraint boundary
    
    # Calculate the dual function value g(λ)
    dual_value = lagrangian(x_min_L, lambda_val)
    
    # Calculate primal optimum
    primal_value = f(constrained_x)
    
    # Plot the objective function
    plt.subplot(2, 2, 1)
    plt.plot(x, f(x), 'b-', linewidth=2, label='$f(x)$')
    plt.axvline(x=constrained_x, color='r', linestyle='--', label='Constrained optimum')
    if optimal_x > min(x) and optimal_x < max(x):
        plt.axvline(x=optimal_x, color='g', linestyle='--', label='Unconstrained optimum')
    plt.axhline(y=primal_value, color='r', linestyle=':')
    plt.grid(True)
    plt.xlim([min(x), max(x)])
    plt.ylim([min(f(x)), max(f(x))])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Objective Function')
    plt.legend()
    
    # Plot the constraint function
    plt.subplot(2, 2, 2)
    plt.plot(x, g(x), 'r-', linewidth=2, label='$g(x)$')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.fill_between(x, g(x), 0, where=(g(x) < 0), color='r', alpha=0.3, label='Feasible region')
    plt.grid(True)
    plt.xlim([min(x), max(x)])
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title('Constraint Function (g(x) ≤ 0)')
    plt.legend()
    
    # Plot the Lagrangian function for the current λ
    plt.subplot(2, 2, 3)
    plt.plot(x, L_x, 'm-', linewidth=2, label=f'$L(x, λ={lambda_val:.2f})$')
    plt.axvline(x=x_min_L, color='g', linestyle='--', label=f'$x^*(λ)={x_min_L:.2f}$')
    plt.axhline(y=dual_value, color='g', linestyle=':')
    plt.grid(True)
    plt.xlim([min(x), max(x)])
    plt.xlabel('x')
    plt.ylabel('L(x, λ)')
    plt.title(f'Lagrangian Function for λ = {lambda_val:.2f}')
    plt.legend()
    
    # Calculate dual function for a range of λ values
    lambda_range = np.linspace(0, 5, 100)
    dual_values = []
    
    for lam in lambda_range:
        if problem_type == 'quadratic':
            x_min = 0 if lam <= 0 else min(0, 2 - lam/2)
        elif problem_type == 'exponential':
            x_min = -np.inf if lam <= 0 else np.log(lam) if lam > 0 else 1
            x_min = min(x_min, 1)
        elif problem_type == 'entropy':
            x_min = np.exp(-lam)
            x_min = max(x_min, 1)
            
        dual_values.append(lagrangian(x_min, lam))
    
    # Plot the dual function
    plt.subplot(2, 2, 4)
    plt.plot(lambda_range, dual_values, 'g-', linewidth=2, label='$g(λ)$')
    plt.axvline(x=lambda_val, color='m', linestyle='--', label=f'Current λ = {lambda_val:.2f}')
    plt.axhline(y=primal_value, color='r', linestyle='--', label='Primal optimum')
    plt.grid(True)
    plt.xlabel('λ')
    plt.ylabel('g(λ)')
    plt.title('Dual Function')
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Display optimization details
    display(Markdown(f"### Problem Details"))
    display(Markdown(f"**Primal Problem:** {title}"))
    display(Markdown(f"**Lagrangian:** $L(x, λ) = f(x) + λ \\cdot g(x)$"))
    display(Markdown(f"**Dual Function:** $g(λ) = \\inf_x L(x, λ)$"))
    display(Markdown(f"**Current λ value:** {lambda_val:.4f}"))
    display(Markdown(f"**Minimizer of L(x, λ):** $x^*(λ) = {x_min_L:.4f}$"))
    display(Markdown(f"**Dual function value:** $g(λ) = {dual_value:.4f}$"))
    display(Markdown(f"**Primal optimal value:** $p^* = {primal_value:.4f}$"))
    display(Markdown(f"**Duality gap:** {max(0, primal_value - dual_value):.4e}"))
    
    # Check for complementary slackness
    cs_value = lambda_val * g(x_min_L)
    display(Markdown(f"**Complementary slackness:** $λ \\cdot g(x^*(λ)) = {cs_value:.4e}$"))
    
    # Economic interpretation
    display(Markdown("### Economic Interpretation"))
    display(Markdown(f"λ represents the shadow price of the constraint. " 
                   f"It tells us how much the optimal value would change " 
                   f"if we relaxed the constraint by one unit."))

# Create interactive widget
def create_interactive_widget():
    problem_dropdown = widgets.Dropdown(
        options=[('Quadratic', 'quadratic'), 
                 ('Exponential', 'exponential'), 
                 ('Entropy', 'entropy')],
        value='quadratic',
        description='Problem:',
    )
    
    lambda_slider = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=5.0,
        step=0.1,
        description='λ:',
        continuous_update=False
    )
    
    interact(plot_lagrangian_duality, 
             problem_type=problem_dropdown,
             lambda_val=lambda_slider)

# Display the interactive widget
if __name__ == "__main__":
    display(Markdown("# Interactive Lagrangian Duality Explorer"))
    display(Markdown("""
    This widget demonstrates how Lagrangian duality works for simple optimization problems.
    
    - Adjust the Lagrange multiplier λ using the slider
    - Choose different problem types from the dropdown
    - Observe how the Lagrangian and dual function change
    """))
    create_interactive_widget()
