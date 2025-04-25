import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------
# 1.  KL divergence and its upper inverse for Bernoulli RVs
# --------------------------------------------------------------------
def kl(p_hat, p):
    """KL( Ber(p_hat) || Ber(p) ), safe at the boundaries."""
    p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))


def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """Upper inverse of the KL-divergence via bisection."""
    eps = np.log(1.0 / delta) / n

    # special cases
    if p_hat <= 0.0:
        return 1.0 - np.exp(-eps)
    if p_hat >= 1.0:
        return 1.0

    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl(p_hat, mid) > eps:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


# --------------------------------------------------------------------
# 2.  Simulation parameters
# --------------------------------------------------------------------
n = 100
delta = 0.05
delta_star = delta / (n + 1)  # makes  ε = ln((n+1)/δ)/n
rng = np.random.default_rng(0)

p_a_grid = np.linspace(0.0, 1.0, 201)  # 0, 0.005, … ,1
gap_vals = []  # p⁺ − p̂ for each p_a
real_dev = []  # actual deviation 0.5 − p̂ (for curiosity)

# --------------------------------------------------------------------
# 3.  Experiment on the grid
# --------------------------------------------------------------------
for p_a in p_a_grid:
    p0 = p1 = (1.0 - p_a) / 2.0
    probs = [p0, p_a, p1]
    support = np.array([0.0, 0.5, 1.0])

    sample = rng.choice(support, size=n, p=probs)
    p_hat = sample.mean()

    p_plus = kl_upper_bound(p_hat, n, delta_star)
    gap_vals.append(p_plus - p_hat)
    real_dev.append(0.5 - p_hat)  # p = 0.5 for all p_a

gap_vals = np.array(gap_vals)
real_dev = np.array(real_dev)

# --------------------------------------------------------------------
# 4.  Plot  gap  versus p_a
# --------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(p_a_grid, gap_vals, label=r"KL bound  $p^{+}-\hat p$")
plt.plot(p_a_grid, real_dev, "--", label=r"Actual error  $p - \hat p$")

plt.xlabel(r"$p_a \; (=P(X=0.5))$")
plt.ylabel(r"Upper bound / realised error")
plt.title(rf"KL upper bounds on $p-\hat p$   ( $n={n}$ ,  $\delta={delta}$ )")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
