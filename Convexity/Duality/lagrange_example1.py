import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the problem
def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return 1 - x[0] - x[1]  # g(x) = 1 - x₁ - x₂ ≥ 0

# Lagrangian function
def lagrangian(x, lambda_val):
    return objective(x) - lambda_val * constraint(x)

# Solve using SciPy (direct approach)
x0 = [0.5, 0.5]
bounds = [(0, None), (0, None)]
constraints = {'type': 'ineq', 'fun': constraint}
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimal solution using direct approach:")
print(f"x* = {result.x}")
print(f"f(x*) = {result.fun}")

# Dual function
def dual_function(lambda_val):
    # For fixed lambda, minimize the Lagrangian over x
    result = minimize(lambda x: lagrangian(x, lambda_val), 
                     x0, 
                     bounds=bounds)
    return result.fun

# Find optimal lambda by maximizing the dual function
lambda_values = np.linspace(0, 5, 100)
dual_values = [dual_function(lambda_val) for lambda_val in lambda_values]

optimal_lambda_idx = np.argmax(dual_values)
optimal_lambda = lambda_values[optimal_lambda_idx]
optimal_dual_value = dual_values[optimal_lambda_idx]

print("\nOptimal solution using dual approach:")
print(f"λ* = {optimal_lambda}")
print(f"g(λ*) = {optimal_dual_value}")

# Solve with fixed optimal lambda
result_dual = minimize(lambda x: lagrangian(x, optimal_lambda), 
                       x0, 
                       bounds=bounds)
print(f"x* = {result_dual.x}")

# Solve using CVXPY for verification
x = cp.Variable(2, nonneg=True)
objective_fn = cp.sum_squares(x)
constraints = [x[0] + x[1] <= 1]
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
problem.solve()

print("\nOptimal solution using CVXPY:")
print(f"x* = {x.value}")
print(f"f(x*) = {problem.value}")
print(f"Optimal dual variables: {constraints[0].dual_value}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, dual_values)
plt.axvline(x=optimal_lambda, color='r', linestyle='--', label=f'Optimal λ = {optimal_lambda:.4f}')
plt.axhline(y=optimal_dual_value, color='g', linestyle='--', label=f'Optimal dual value = {optimal_dual_value:.4f}')
plt.axhline(y=problem.value, color='b', linestyle='--', label=f'Primal optimal value = {problem.value:.4f}')
plt.xlabel('λ')
plt.ylabel('g(λ)')
plt.title('Dual Function')
plt.legend()
plt.grid(True)
plt.savefig('dual_function_plot.png')
plt.close()

# Create a contour plot to visualize the optimization problem
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2

# Create the constraint line x1 + x2 = 1
x1_line = np.linspace(0, 1, 100)
x2_line = 1 - x1_line

plt.figure(figsize=(10, 8))
cp = plt.contour(X1, X2, Z, levels=20)
plt.colorbar(cp, label='f(x) = x₁² + x₂²')
plt.plot(x1_line, x2_line, 'r-', label='x₁ + x₂ = 1')
plt.fill_between(x1_line, 0, x2_line, alpha=0.2, color='r', label='Feasible region')
plt.plot(result.x[0], result.x[1], 'ko', markersize=10, label='Optimal solution')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Constrained Optimization Problem')
plt.legend()
plt.grid(True)
plt.savefig('optimization_contour.png')
plt.close()
