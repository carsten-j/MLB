import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar


def compute_conjugate(f, lambda_val, x_range=(-10, 10), method="bounded"):
    """
    Compute the conjugate function value f*(λ) for a given function f.

    Parameters:
    -----------
    f : function
        The original function f(x)
    lambda_val : float
        The value of λ at which to compute the conjugate
    x_range : tuple
        The range of x values to consider for the optimization
    method : str
        Optimization method for scipy.optimize.minimize_scalar

    Returns:
    --------
    conj_val : float
        The conjugate function value f*(λ)
    optimal_x : float
        The x value that achieves the supremum
    """

    # Define the objective function to maximize: λx - f(x)
    def objective(x):
        return lambda_val * x - f(x)

    # Use the negative for minimization (since we want to maximize)
    def neg_objective(x):
        return -objective(x)

    # Find the x that maximizes λx - f(x)
    result = minimize_scalar(neg_objective, bounds=x_range, method=method)
    optimal_x = result.x
    conj_val = objective(optimal_x)

    return conj_val, optimal_x


def plot_function_and_conjugate(f, title="Function and its Conjugate"):
    """
    Plot a function and its conjugate function.

    Parameters:
    -----------
    f : function
        The original function f(x)
    title : str
        The title for the plot
    """
    # Create a figure with a 2x2 grid layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])

    # Subplot for the original function f(x)
    ax_func = fig.add_subplot(gs[0, 0])

    # Subplot for the conjugate function f*(λ)
    ax_conj = fig.add_subplot(gs[1, 0])

    # Subplot for the visualization of conjugate pairs
    ax_visual = fig.add_subplot(gs[:, 1])

    # Plot the original function
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = [f(x) for x in x_vals]
    ax_func.plot(x_vals, y_vals, "b-", label="f(x)")
    ax_func.set_title("Original Function f(x)")
    ax_func.set_xlabel("x")
    ax_func.set_ylabel("f(x)")
    ax_func.grid(True, alpha=0.3)
    ax_func.legend()

    # Compute and plot the conjugate function
    lambda_vals = np.linspace(-3, 3, 100)
    conj_vals = []
    optimal_xs = []

    for lam in lambda_vals:
        conj_val, opt_x = compute_conjugate(f, lam)
        conj_vals.append(conj_val)
        optimal_xs.append(opt_x)

    ax_conj.plot(lambda_vals, conj_vals, "g-", label="f*(λ)")
    ax_conj.set_title("Conjugate Function f*(λ)")
    ax_conj.set_xlabel("λ")
    ax_conj.set_ylabel("f*(λ)")
    ax_conj.grid(True, alpha=0.3)
    ax_conj.legend()

    # Visualization of conjugate pairs
    # Select a few λ values for visualization
    selected_lambdas = [-2, -1, 0, 1, 2]
    colors = ["r", "m", "c", "y", "k"]

    for lam, color in zip(selected_lambdas, colors):
        # Find the conjugate value and optimal x for this λ
        idx = np.abs(lambda_vals - lam).argmin()
        conj_val = conj_vals[idx]
        opt_x = optimal_xs[idx]

        # Plot tangent line on the original function
        tangent_y = [lam * x - conj_val for x in x_vals]
        ax_func.plot(x_vals, tangent_y, f"{color}--", alpha=0.5)

        # Mark the tangent point
        func_val = f(opt_x)
        ax_func.plot([opt_x], [func_val], f"{color}o", markersize=5)

        # Draw a point on the conjugate function
        ax_conj.plot([lam], [conj_val], f"{color}o", markersize=5)

        # Visualize the conjugate relationship
        ax_visual.plot([opt_x], [lam], f"{color}o", label=f"λ={lam}")
        ax_visual.plot([opt_x], [lam], f"{color}+", markersize=8)

    ax_visual.set_title("Conjugate Pairs (x, λ)")
    ax_visual.set_xlabel("x (primal coordinate)")
    ax_visual.set_ylabel("λ (dual coordinate)")
    ax_visual.grid(True, alpha=0.3)
    ax_visual.legend()

    plt.tight_layout()
    plt.savefig("function_and_conjugate.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Example function: f(x) = x^2*(x-1)*(x+1) = x^4 - x^2
    # f'(x) = 4x**3 - 2x
    def example_function(x):
        return x**2 * (x - 1) * (x + 1)

    # Alternative representation to verify:
    # def example_function(x):
    #     return x**4 - x**2

    plot_function_and_conjugate(
        example_function, title="Function f(x) = x²(x-1)(x+1) and its Conjugate"
    )

    # Print some conjugate values for specific slopes
    print("Conjugate function values for selected λ:")
    for lam in [-2, -1, 0, 1, 2]:
        conj_val, opt_x = compute_conjugate(example_function, lam)
        print(f"f*({lam}) = {conj_val:.6f} (achieved at x = {opt_x:.6f})")
