import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

'''
Constrained Logistic Regression with L1 Budget Constraint

Primal problem:
    minimize    sum(log(1 + exp(-y_i(w^T x_i + b))))
    subject to  ||w||_1 <= t

We'll solve this using Lagrangian duality and also compare with direct CVXPY solution
'''

# Generate data for binary classification
X, y = datasets.load_breast_cancer(return_X_y=True)

# Convert labels to -1/+1
y = 2 * y - 1

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_samples, n_features = X_train.shape

# Solve using CVXPY (primal formulation)
w = cp.Variable(n_features)
b = cp.Variable()

# Logistic loss
log_likelihood = cp.sum(cp.logistic(cp.multiply(-y_train, X_train @ w + b)))

# Budget parameter
t = 5.0

# L1 constraint
l1_constraint = [cp.norm(w, 1) <= t]

# Solve the constrained problem
primal_problem = cp.Problem(cp.Minimize(log_likelihood), l1_constraint)
primal_problem.solve()

print("Primal solution:")
print(f"Objective value: {primal_problem.value}")
print(f"L1 norm of weights: {cp.norm(w, 1).value}")
print(f"Number of non-zero weights: {np.sum(np.abs(w.value) > 1e-4)}")

# Now implement the dual approach
# The Lagrangian is: L(w, b, λ) = log_loss + λ(||w||_1 - t)

def objective_function(w, b, X, y):
    """Logistic regression objective"""
    z = y * (X @ w + b)
    # Numerical stability for log(1 + exp(-z))
    loss = np.zeros_like(z)
    idx_pos = z > 0
    loss[idx_pos] = np.log(1 + np.exp(-z[idx_pos]))
    loss[~idx_pos] = -z[~idx_pos] + np.log(1 + np.exp(z[~idx_pos]))
    return np.mean(loss)

def dual_function(lambda_val, X, y, t):
    """
    Evaluate the dual function at a given lambda.
    For each λ, we need to minimize the Lagrangian over w and b.
    """
    # For fixed λ, solve the unconstrained problem
    # min_w,b log_loss + λ||w||_1
    w = cp.Variable(n_features)
    b = cp.Variable()
    
    objective = cp.sum(cp.logistic(cp.multiply(-y, X @ w + b))) + lambda_val * cp.norm(w, 1)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    
    # The dual function value
    return problem.value - lambda_val * t, w.value, b.value

# Evaluate the dual function for different λ values
lambda_values = np.linspace(0, 0.5, 50)
dual_values = []
w_values = []
b_values = []
l1_norms = []
sparsity = []

for lam in lambda_values:
    d_val, w_val, b_val = dual_function(lam, X_train, y_train, t)
    dual_values.append(d_val)
    w_values.append(w_val)
    b_values.append(b_val)
    l1_norms.append(np.sum(np.abs(w_val)))
    sparsity.append(np.sum(np.abs(w_val) > 1e-4))

# Find the optimal λ that makes ||w||_1 close to t
optimal_idx = np.argmin(np.abs(np.array(l1_norms) - t))
optimal_lambda = lambda_values[optimal_idx]
optimal_w = w_values[optimal_idx]
optimal_b = b_values[optimal_idx]

print("\nDual solution:")
print(f"Optimal λ: {optimal_lambda}")
print(f"L1 norm at optimal λ: {l1_norms[optimal_idx]}")
print(f"Number of non-zero weights: {sparsity[optimal_idx]}")

# Evaluate objective function at the optimal w, b from dual
obj_val = objective_function(optimal_w, optimal_b, X_train, y_train)
print(f"Objective value at optimal dual solution: {obj_val}")

# Calculate the duality gap
duality_gap = obj_val - primal_problem.value
print(f"Duality gap: {duality_gap}")

# Plot the dual function
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, dual_values)
plt.axvline(x=optimal_lambda, color='r', linestyle='--', 
            label=f'Optimal λ = {optimal_lambda:.4f}')
plt.xlabel('λ')
plt.ylabel('g(λ)')
plt.title('Dual Function for Constrained Logistic Regression')
plt.legend()
plt.grid(True)
plt.savefig('logistic_regression_dual.png')

# Plot L1 norm as a function of λ
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, l1_norms)
plt.axhline(y=t, color='g', linestyle='--', label=f'Budget t = {t}')
plt.axvline(x=optimal_lambda, color='r', linestyle='--')
plt.xlabel('λ')
plt.ylabel('||w||_1')
plt.title('L1 Norm vs. Lagrange Multiplier')
plt.legend()
plt.grid(True)
plt.savefig('l1_norm_vs_lambda.png')

# Plot sparsity as a function of λ
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, sparsity)
plt.axvline(x=optimal_lambda, color='r', linestyle='--')
plt.xlabel('λ')
plt.ylabel('Number of non-zero weights')
plt.title('Sparsity vs. Lagrange Multiplier')
plt.grid(True)
plt.savefig('sparsity_vs_lambda.png')

# Compare test accuracy
def predict(X, w, b):
    return np.sign(X @ w + b)

# Test accuracy with primal solution
y_pred_primal = predict(X_test, w.value, b.value)
accuracy_primal = accuracy_score(y_test, y_pred_primal)

# Test accuracy with dual solution
y_pred_dual = predict(X_test, optimal_w, optimal_b)
accuracy_dual = accuracy_score(y_test, y_pred_dual)

print("\nTest accuracy comparison:")
print(f"Primal solution accuracy: {accuracy_primal * 100:.2f}%")
print(f"Dual solution accuracy: {accuracy_dual * 100:.2f}%")

# Visualize weight differences between primal and dual solutions
plt.figure(figsize=(12, 6))
plt.bar(range(n_features), w.value, alpha=0.5, label='Primal weights')
plt.bar(range(n_features), optimal_w, alpha=0.5, label='Dual weights')
plt.xlabel('Feature index')
plt.ylabel('Weight value')
plt.title('Comparison of Weight Values: Primal vs Dual')
plt.legend()
plt.grid(True)
plt.savefig('weight_comparison.png')

# Feature selection insight - identify the most important features
primal_important = np.argsort(np.abs(w.value))[-10:]
dual_important = np.argsort(np.abs(optimal_w))[-10:]

print("\nTop 10 important features:")
print(f"Primal: {primal_important}")
print(f"Dual: {dual_important}")
print(f"Features in common: {np.intersect1d(primal_important, dual_important)}")

# Economic interpretation of the Lagrange multiplier
print("\nEconomic interpretation:")
print(f"The Lagrange multiplier λ = {optimal_lambda} represents the shadow price of the L1 budget constraint.")
print(f"It tells us that increasing the L1 budget by 1 unit would decrease the objective by approximately {optimal_lambda} units.")

# Verify complementary slackness: λ(||w||_1 - t) = 0
complementary_slackness = optimal_lambda * (l1_norms[optimal_idx] - t)
print(f"\nComplementary slackness: λ(||w||_1 - t) = {complementary_slackness}")
print("This should be close to zero if complementary slackness holds.")
