import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Example quasiconvex functions for optimization

def example_qcvx_1d(x):
    """A quasiconvex function: f(x) = |x - 2| + |x + 2| - 3"""
    return np.abs(x - 2) + np.abs(x + 2) - 3

def example_qcvx_2d(x):
    """A 2D quasiconvex function"""
    return np.maximum(np.abs(x[0] - 1), np.abs(x[1] + 2))

def example_qcvx_not_convex_1d(x):
    """A quasiconvex function that is not convex"""
    return -1 / (1 + (x - 1)**2)

def example_qcvx_not_convex_2d(x):
    """A 2D quasiconvex function that is not convex"""
    # f(x,y) = -1/sqrt(1 + (x-1)^2 + (y+1)^2)
    return -1 / np.sqrt(1 + (x[0] - 1)**2 + (x[1] + 1)**2)

# Function to visualize 1D optimization
def visualize_1d_optimization(func, x_min, x_max, result):
    x = np.linspace(x_min, x_max, 1000)
    y = [func(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label=f'Function')
    plt.scatter([result.x[0]], [result.fun], color='r', s=100, 
                label=f'Minimum at x={result.x[0]:.4f}, f(x)={result.fun:.4f}')
    
    plt.title('1D Quasiconvex Optimization')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig('quasiconvex_optimization_1d.png')

# Function to visualize 2D optimization
def visualize_2d_optimization(func, x_min, x_max, y_min, y_max, result):
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = func([X[j, i], Y[j, i]])
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(211, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter([result.x[0]], [result.x[1]], [result.fun], color='r', s=100, label='Minimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D Surface Plot of Function')
    
    # Contour plot
    ax2 = fig.add_subplot(212)
    contour = ax2.contour(X, Y, Z, 20, cmap='viridis')
    ax2.scatter([result.x[0]], [result.x[1]], color='r', s=100, label='Minimum')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Optimization Path')
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('quasiconvex_optimization_2d.png')

# Quasiconvex optimization using different starting points
def optimize_from_multiple_starts(func, bounds, n_starts=5):
    best_result = None
    best_value = float('inf')
    
    results = []
    start_points = []
    
    # Generate random starting points within bounds
    for _ in range(n_starts):
        if len(bounds) == 1:  # 1D case
            x0 = np.random.uniform(bounds[0][0], bounds[0][1])
            start_points.append([x0])
        else:  # Multi-dimensional case
            x0 = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
            start_points.append(x0)
        
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        results.append(result)
        
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
    
    return best_result, results, start_points

if __name__ == "__main__":
    print("Optimization Examples for Quasiconvex Functions")
    
    # 1D optimization example for standard quasiconvex function
    print("\n1. Optimizing 1D quasiconvex function: |x - 2| + |x + 2| - 3")
    bounds_1d = [(-5, 5)]
    result_1d, all_results_1d, start_points_1d = optimize_from_multiple_starts(example_qcvx_1d, bounds_1d)
    print(f"Global minimum found at x = {result_1d.x[0]:.6f}, with value = {result_1d.fun:.6f}")
    visualize_1d_optimization(example_qcvx_1d, -5, 5, result_1d)
    
    # 1D optimization example for non-convex quasiconvex function
    print("\n2. Optimizing 1D quasiconvex (but non-convex) function: -1/(1+(x-1)²)")
    result_1d_nc, all_results_1d_nc, start_points_1d_nc = optimize_from_multiple_starts(
        example_qcvx_not_convex_1d, bounds_1d
    )
    print(f"Global minimum found at x = {result_1d_nc.x[0]:.6f}, with value = {result_1d_nc.fun:.6f}")
    visualize_1d_optimization(example_qcvx_not_convex_1d, -5, 5, result_1d_nc)
    
    # 2D optimization example
    print("\n3. Optimizing 2D quasiconvex function: max(|x-1|, |y+2|)")
    bounds_2d = [(-5, 5), (-5, 5)]
    result_2d, all_results_2d, start_points_2d = optimize_from_multiple_starts(example_qcvx_2d, bounds_2d)
    print(f"Global minimum found at (x,y) = ({result_2d.x[0]:.6f}, {result_2d.x[1]:.6f}), with value = {result_2d.fun:.6f}")
    visualize_2d_optimization(example_qcvx_2d, -5, 5, -5, 5, result_2d)
    
    # 2D optimization example for non-convex quasiconvex function
    print("\n4. Optimizing 2D quasiconvex (but non-convex) function: -1/sqrt(1+(x-1)²+(y+1)²)")
    result_2d_nc, all_results_2d_nc, start_points_2d_nc = optimize_from_multiple_starts(
        example_qcvx_not_convex_2d, bounds_2d
    )
    print(f"Global minimum found at (x,y) = ({result_2d_nc.x[0]:.6f}, {result_2d_nc.x[1]:.6f}), with value = {result_2d_nc.fun:.6f}")
    visualize_2d_optimization(example_qcvx_not_convex_2d, -5, 5, -5, 5, result_2d_nc)
    
    print("\nComparison of results from different starting points:")
    print("\nFor 1D quasiconvex function:")
    for i, (result, start) in enumerate(zip(all_results_1d, start_points_1d)):
        print(f"  Start: {start[0]:.4f}, Minimum: x = {result.x[0]:.6f}, f(x) = {result.fun:.6f}")
    
    print("\nFor 2D quasiconvex function:")
    for i, (result, start) in enumerate(zip(all_results_2d, start_points_2d)):
        print(f"  Start: ({start[0]:.4f}, {start[1]:.4f}), Minimum: (x,y) = ({result.x[0]:.6f}, {result.x[1]:.6f}), f(x,y) = {result.fun:.6f}")
