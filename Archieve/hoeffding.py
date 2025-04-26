import matplotlib.pyplot as plt
import numpy as np

# Parameters
n = 1000  # sample size
delta = 0.01  # confidence level

# Grid of empirical averages  \hat p_n  in [0,1]
p_hat = np.linspace(0.0, 1.0, 500)

# Hoeffding error term  \epsilon = sqrt( ln(1/δ) / (2n) )
epsilon = np.sqrt(np.log(1.0 / delta) / (2.0 * n))

# Upper‐bound on the true bias  p  (may exceed 1 slightly; clip at 1 for display)
upper_bound = np.clip(p_hat + epsilon, 0.0, 1.0)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(
    p_hat,
    upper_bound,
    label=r"Hoeffding bound:  $\hat p_n + \sqrt{\ln(1/\delta)/(2n)}$",
)
plt.plot(p_hat, p_hat, "k--", label=r"identity:  $p=\hat p_n$")

plt.title(f"Hoeffding upper bound (n={n},  δ={delta})")
plt.xlabel(r"Empirical average  $\hat p_n$")
plt.ylabel(r"Upper bound on $p$")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
