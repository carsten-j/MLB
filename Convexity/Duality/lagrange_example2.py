import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

'''
Resource Allocation Problem:
Maximize total utility: U(x) = sum(log(x_i)) where x_i is resource allocated to task i
Subject to: sum(x_i) <= total_resources
           x_i >= 0 for all i

This models a fair resource allocation problem with diminishing returns.
The logarithmic utility function ensures fairness (allocating some resources to each task).
'''

# Problem parameters
n_tasks = 5  # Number of tasks
total_resources = 10  # Total resources available

# Solve using CVXPY
x = cp.Variable(n_tasks, nonneg=True)
objective = cp.sum(cp.log(x))  # Logarithmic utility function
constraints = [cp.sum(x) <= total_resources]
problem = cp.Problem(cp.Maximize(objective), constraints)
problem.solve()

print("Optimal allocation using CVXPY:")
print(f"x* = {x.value}")
print(f"Total utility = {problem.value}")
print(f"Optimal dual variable (resource shadow price) = {constraints[0].dual_value}")

# Let's manually implement and visualize the Lagrangian dual approach
# The Lagrangian is: L(x, λ) = sum(log(x_i)) - λ(sum(x_i) - total_resources)

def optimal_allocation(lambda_val):
    """
    For a given lambda (dual variable), find the optimal allocation.
    From KKT conditions, we know that the optimal allocation is:
    x_i = 1/λ for all i
    """
    if lambda_val <= 0:
        return np.ones(n_tasks) * float('inf')  # Unbounded
    return np.ones(n_tasks) / lambda_val

def dual_function(lambda_val):
    """
    Evaluate the dual function at a given lambda.
    g(λ) = max_x L(x, λ) = n*log(1/λ) + λ*total_resources
    """
    if lambda_val <= 0:
        return float('inf')  # Unbounded
    x_opt = optimal_allocation(lambda_val)
    utility = np.sum(np.log(x_opt))
    return utility + lambda_val * (total_resources - np.sum(x_opt))

# Visualize the dual function
lambda_values = np.linspace(0.01, 2, 100)
dual_values = [dual_function(lam) for lam in lambda_values]

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, dual_values)
plt.axvline(x=constraints[0].dual_value, color='r', linestyle='--', 
            label=f'Optimal λ = {constraints[0].dual_value:.4f}')
plt.xlabel('λ')
plt.ylabel('g(λ)')
plt.title('Dual Function for Resource Allocation Problem')
plt.legend()
plt.grid(True)
plt.savefig('resource_allocation_dual.png')

# Calculate optimal allocation with the optimal lambda
optimal_lambda = constraints[0].dual_value
x_optimal = optimal_allocation(optimal_lambda)

print("\nOptimal allocation using dual approach:")
print(f"x* = {x_optimal}")
print(f"Sum of allocations = {np.sum(x_optimal)}")

# Visualize the allocations
plt.figure(figsize=(10, 6))
plt.bar(range(n_tasks), x.value, color='skyblue', label='CVXPY Solution')
plt.bar(range(n_tasks), x_optimal, color='none', edgecolor='red', linewidth=2, 
        label='Dual Solution')
plt.axhline(y=total_resources/n_tasks, color='g', linestyle='--', 
            label=f'Equal allocation ({total_resources/n_tasks:.2f})')
plt.xlabel('Task')
plt.ylabel('Resource Allocation')
plt.title('Optimal Resource Allocation')
plt.legend()
plt.grid(True, axis='y')
plt.savefig('resource_allocation.png')

# Complementary slackness verification
print("\nComplementary slackness verification:")
print(f"λ*(sum(x*) - total_resources) = {optimal_lambda * (np.sum(x_optimal) - total_resources)}")
print("Should be close to zero if complementary slackness holds")

# Economic interpretation
print("\nEconomic interpretation:")
print(f"Shadow price of resource = {optimal_lambda}")
print(f"This means adding one more unit of resource would increase total utility by ~{optimal_lambda}")

# Verify this by solving with slightly more resources
problem_more_resources = cp.Problem(
    cp.Maximize(cp.sum(cp.log(x))), 
    [cp.sum(x) <= total_resources + 1]
)
problem_more_resources.solve()
print(f"Utility with {total_resources} resources: {problem.value}")
print(f"Utility with {total_resources + 1} resources: {problem_more_resources.value}")
print(f"Difference: {problem_more_resources.value - problem.value}")
