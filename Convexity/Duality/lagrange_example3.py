import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
Support Vector Machines (SVMs) are a perfect example of Lagrangian duality in practice.
The primal SVM problem is:
    minimize (1/2)||w||^2
    subject to y_i(w^T x_i + b) >= 1 for all i

The dual problem becomes:
    maximize sum(α_i) - (1/2)sum(α_i α_j y_i y_j x_i^T x_j)
    subject to α_i >= 0 for all i
               sum(α_i y_i) = 0
               
This dual formulation leads to the kernel trick, one of the most powerful ideas in machine learning.
'''

# Generate synthetic data for binary classification
X, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42, 
                                   n_clusters_per_class=1)
# Convert labels to -1/+1
y = 2 * y - 1

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM using the primal formulation with CVXPY
n_samples, n_features = X_train.shape

# Variables
w = cp.Variable(n_features)
b = cp.Variable()
xi = cp.Variable(n_samples)  # Slack variables for soft margin
C = 1.0  # Regularization parameter

# Objective: minimize ||w||^2/2 + C*sum(xi)
objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))

# Constraints: y_i(w^T x_i + b) >= 1 - xi_i, xi_i >= 0
constraints = [
    cp.multiply(y_train, X_train @ w + b) >= 1 - xi,
    xi >= 0
]

# Solve the problem
primal_problem = cp.Problem(objective, constraints)
primal_problem.solve()

print("Primal SVM solution:")
print(f"w* = {w.value}")
print(f"b* = {b.value}")
print(f"Objective value = {primal_problem.value}")

# Implement the dual formulation
# Variables: alpha (Lagrange multipliers)
alpha = cp.Variable(n_samples, nonneg=True)

# Gram matrix
K = X_train @ X_train.T  # Linear kernel K(x_i, x_j) = x_i^T x_j
y_outer = np.outer(y_train, y_train)
Q = K * y_outer  # Element-wise product

# Dual objective: maximize sum(alpha_i) - (1/2)sum(alpha_i alpha_j y_i y_j K(x_i, x_j))
dual_objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, Q))

# Dual constraints: 0 <= alpha_i <= C, sum(alpha_i y_i) = 0
dual_constraints = [
    alpha <= C,
    cp.sum(cp.multiply(alpha, y_train)) == 0
]

dual_problem = cp.Problem(dual_objective, dual_constraints)
dual_problem.solve()

print("\nDual SVM solution:")
print(f"Objective value = {dual_problem.value}")
print(f"Strong duality gap = {primal_problem.value + dual_problem.value}")

# Recover primal variables from dual solution
# w = sum(alpha_i * y_i * x_i)
w_dual = np.sum(alpha.value[:, np.newaxis] * y_train[:, np.newaxis] * X_train, axis=0)

# Find support vectors (points where alpha_i > 0)
sv_indices = np.where(alpha.value > 1e-5)[0]
support_vectors = X_train[sv_indices]
sv_y = y_train[sv_indices]
sv_alphas = alpha.value[sv_indices]

# Calculate b using support vectors
b_dual = np.mean(sv_y - np.dot(support_vectors, w_dual))

print("\nRecovered primal solution from dual:")
print(f"w_dual = {w_dual}")
print(f"b_dual = {b_dual}")
print(f"||w - w_dual|| = {np.linalg.norm(w.value - w_dual)}")
print(f"Number of support vectors: {len(sv_indices)} out of {n_samples} points")

# Visualize the decision boundary and support vectors
def plot_decision_boundary(w, b, X, y, support_vectors=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Class -1')
    
    # Plot support vectors
    if support_vectors is not None:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, 
                   facecolors='none', edgecolors='g', linewidth=2, label='Support Vectors')
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='k', linestyles='--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary and Support Vectors')
    plt.legend()
    plt.grid(True)
    plt.savefig('svm_decision_boundary.png')
    plt.close()

# Plot the decision boundary using the dual solution
plot_decision_boundary(w_dual, b_dual, X_train, y_train, support_vectors)

# KKT conditions verification
print("\nKKT conditions verification:")

# 1. Primal feasibility
margin_values = y_train * (X_train @ w_dual + b_dual)
print(f"Min margin value: {np.min(margin_values)}")
if np.min(margin_values) >= -1e-5:
    print("Primal feasibility satisfied ✓")
else:
    print("Primal feasibility violated ✗")

# 2. Dual feasibility (alpha_i >= 0)
if np.min(alpha.value) >= -1e-5:
    print("Dual feasibility satisfied ✓")
else:
    print("Dual feasibility violated ✗")

# 3. Complementary slackness: alpha_i * (y_i(w^T x_i + b) - 1) = 0
slackness = alpha.value * (margin_values - 1)
print(f"Max complementary slackness value: {np.max(np.abs(slackness))}")
if np.max(np.abs(slackness)) < 1e-5:
    print("Complementary slackness satisfied ✓")
else:
    print("Complementary slackness violated ✗")

# Test accuracy
def predict(X, w, b):
    return np.sign(X @ w + b)

y_pred = predict(X_test, w_dual, b_dual)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
