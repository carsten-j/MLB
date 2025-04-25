import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# Example quasiconvex functions
def example1(x):
    """A simple quasiconvex function: f(x) = |x|"""
    return np.abs(x)

def example2(x):
    """A quasiconvex function that is not convex: f(x) = 1/(1+x^2)"""
    return 1 / (1 + x**2)

def example3(x):
    """A more complex quasiconvex function"""
    return np.abs(x - 2) + np.abs(x + 1) - 2

def example4(x):
    """A quasiconvex function with multiple local minima in its derivative"""
    return np.abs(x) + 0.1 * np.sin(10 * x)

# Bisection method for quasiconvex function optimization
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for optimizing a quasiconvex function.
    
    Parameters:
    -----------
    func : callable
        The quasiconvex function to minimize
    a, b : float
        Initial interval bounds
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    x_min : float
        The approximated minimizer
    f_min : float
        The minimum function value
    iterations : int
        Number of iterations performed
    history : list
        History of intervals and function evaluations
    """
    if a > b:
        a, b = b, a
    
    history = []
    
    for i in range(max_iter):
        # Calculate midpoint
        m = (a + b) / 2
        
        # Calculate points for evaluation
        x1 = m - tol/2
        x2 = m + tol/2
        
        # Evaluate function at these points
        f1 = func(x1)
        f2 = func(x2)
        
        # Record current state
        history.append({
            'iteration': i,
            'interval': [a, b],
            'midpoint': m,
            'x1': x1,
            'x2': x2,
            'f1': f1,
            'f2': f2
        })
        
        # Update interval according to function values
        if f1 < f2:
            b = x2
        elif f1 > f2:
            a = x1
        else:
            # Equal values, shrink from both sides
            a = x1
            b = x2
        
        # Check for convergence
        if (b - a) < tol:
            break
    
    # Return midpoint of final interval
    x_min = (a + b) / 2
    f_min = func(x_min)
    
    return x_min, f_min, i + 1, history

# Golden section search method for comparison
def golden_section(func, a, b, tol=1e-6, max_iter=100):
    """
    Golden section search method for function minimization.
    
    Parameters:
    -----------
    func : callable
        The function to minimize
    a, b : float
        Initial interval bounds
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    x_min : float
        The approximated minimizer
    f_min : float
        The minimum function value
    iterations : int
        Number of iterations performed
    history : list
        History of intervals and function evaluations
    """
    golden_ratio = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    
    if a > b:
        a, b = b, a
    
    # Initial points
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    
    fc = func(c)
    fd = func(d)
    
    history = []
    
    for i in range(max_iter):
        history.append({
            'iteration': i,
            'interval': [a, b],
            'c': c,
            'd': d,
            'fc': fc,
            'fd': fd
        })
        
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = func(d)
        
        if abs(b - a) < tol:
            break
    
    x_min = (a + b) / 2
    f_min = func(x_min)
    
    return x_min, f_min, i + 1, history

# Binary search for λ-sublevel sets
def binary_search_sublevel(func, level, a, b, tol=1e-6, max_iter=100):
    """
    Binary search to find the boundary of a λ-sublevel set.
    For a quasiconvex function, the sublevel sets are convex.

    Parameters:
    -----------
    func : callable
        The quasiconvex function
    level : float
        The level λ for the sublevel set
    a, b : float
        Initial interval bounds (f(a) ≤ λ < f(b))
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    boundary : float
        The approximated boundary of the sublevel set
    iterations : int
        Number of iterations performed
    """
    # Ensure a is in the sublevel set and b is outside
    if func(a) > level:
        if func(b) <= level:
            a, b = b, a
        else:
            raise ValueError("Neither bound is in the λ-sublevel set")
    
    iterations = 0
    
    for i in range(max_iter):
        iterations += 1
        
        # Calculate midpoint
        mid = (a + b) / 2
        f_mid = func(mid)
        
        # Check if midpoint is in sublevel set
        if f_mid <= level:
            a = mid
        else:
            b = mid
        
        # Check for convergence
        if (b - a) < tol:
            break
    
    return (a + b) / 2, iterations

# Visualize the optimization process
def visualize_optimization(func, method_name, result, x_min, x_max):
    x_min, f_min, iterations, history = result
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the function
    x = np.linspace(x_min, x_max, 1000)
    y = [func(xi) for xi in x]
    ax.plot(x, y, 'b-', label=f'Function')
    
    # Plot the final minimum
    ax.scatter([x_min], [f_min], color='red', s=100, 
                label=f'Minimum: x={x_min:.6f}, f(x)={f_min:.6f}')
    
    # Plot the bisection intervals
    if method_name == "Bisection":
        for i, step in enumerate(history[::max(1, len(history)//10)]):  # Plot every 10th step
            a, b = step['interval']
            height = func(a) * 0.97  # Slightly below the function value for visibility
            ax.plot([a, b], [height, height], 'g-', alpha=0.3)
            if i == 0:
                ax.text(a, height, 'Initial', fontsize=8, ha='center')
                ax.text(b, height, 'Interval', fontsize=8, ha='center')
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{method_name} Method: Found minimum in {iterations} iterations')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    plt.savefig(f'quasiconvex_{method_name.lower()}_optimization.png')
    plt.close()

# Compare different optimization methods
def compare_methods(funcs, x_min, x_max, tol=1e-6, max_iter=100):
    results = {}
    
    for name, func in funcs.items():
        print(f"\nOptimizing {name}:")
        
        # Find true minimum using scipy's minimize for comparison
        true_result = minimize(func, x0=0, bounds=[(x_min, x_max)], method='L-BFGS-B')
        true_x = true_result.x[0]
        true_f = true_result.fun
        print(f"True minimum (using scipy): x = {true_x:.6f}, f(x) = {true_f:.6f}")
        
        # Bisection method
        start_time = time.time()
        bisection_result = bisection_method(func, x_min, x_max, tol, max_iter)
        bisection_time = time.time() - start_time
        
        # Golden section method
        start_time = time.time()
        golden_result = golden_section(func, x_min, x_max, tol, max_iter)
        golden_time = time.time() - start_time
        
        # Print results
        print(f"Bisection method: x = {bisection_result[0]:.6f}, f(x) = {bisection_result[1]:.6f}, iterations = {bisection_result[2]}, time = {bisection_time:.6f}s")
        print(f"Golden section: x = {golden_result[0]:.6f}, f(x) = {golden_result[1]:.6f}, iterations = {golden_result[2]}, time = {golden_time:.6f}s")
        
        # Store results
        results[name] = {
            'true': (true_x, true_f),
            'bisection': bisection_result,
            'bisection_time': bisection_time,
            'golden': golden_result,
            'golden_time': golden_time
        }
        
        # Visualize
        visualize_optimization(func, "Bisection", bisection_result, x_min, x_max)
        visualize_optimization(func, "Golden Section", golden_result, x_min, x_max)
    
    return results

# Visualize sublevel sets for quasiconvex functions
def visualize_sublevel_sets(func, x_min, x_max, levels):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    x = np.linspace(x_min, x_max, 1000)
    y = [func(xi) for xi in x]
    ax.plot(x, y, 'b-', label=f'Function')
    
    # Plot sublevel sets
    for level in levels:
        # Find all x values where f(x) <= level
        sublevel_indices = [i for i, yi in enumerate(y) if yi <= level]
        sublevel_x = [x[i] for i in sublevel_indices]
        sublevel_y = [level] * len(sublevel_x)
        
        # Plot the sublevel set
        ax.plot(sublevel_x, sublevel_y, '-', label=f'λ = {level:.2f}')
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Sublevel Sets for Quasiconvex Function')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    plt.savefig('quasiconvex_sublevel_sets.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Define functions to test
    functions = {
        "Example 1: |x|": example1,
        "Example 2: 1/(1+x²)": example2,
        "Example 3: |x-2| + |x+1| - 2": example3,
        "Example 4: |x| + 0.1*sin(10x)": example4
    }
    
    # Compare optimization methods
    results = compare_methods(functions, -5.0, 5.0, 1e-6, 100)
    
    # Visualize sublevel sets for one function
    print("\nVisualizing sublevel sets for Example 2")
    visualize_sublevel_sets(example2, -5.0, 5.0, [0.1, 0.2, 0.3, 0.5, 1.0])
    
    # Demonstrate binary search for sublevel sets
    print("\nDemonstrating binary search for sublevel set boundaries:")
    func = example2
    level = 0.2
    
    # Find left boundary
    left_boundary, left_iters = binary_search_sublevel(func, level, 0, -5, 1e-6, 100)
    print(f"Left boundary of {level}-sublevel set: x = {left_boundary:.6f}, found in {left_iters} iterations")
    
    # Find right boundary
    right_boundary, right_iters = binary_search_sublevel(func, level, 0, 5, 1e-6, 100)
    print(f"Right boundary of {level}-sublevel set: x = {right_boundary:.6f}, found in {right_iters} iterations")
    
    print(f"\nThe {level}-sublevel set is approximately the interval [{left_boundary:.6f}, {right_boundary:.6f}]")
