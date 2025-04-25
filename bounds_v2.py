import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# 1. KL divergence for Bernoulli distributions
# ---------------------------------------------------------------------
def kl(p_hat, p):
    """
    Calculate KL divergence between two Bernoulli distributions when
    p_hat or p in {0,1}

    Parameters:
    p_hat : float
        Probability parameter for the true distribution (P)
    p : float
        Probability parameter for the approximate distribution (Q)

    Returns:
    float or numpy.ndarray
        KL(Q || P)
    """

    # clip to keep log well–defined
    p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))


# ---------------------------------------------------------------------
# 2. Upper inverse of the KL divergence
# ---------------------------------------------------------------------
def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """
    Return p⁺ such that  KL(p_hat||p⁺) = ln(1/delta)/n  and  p⁺ >= p_hat.
    Probability guarantee:  P(p > p⁺) <= delta.
    """
    eps = np.log(1.0 / delta) / n

    # Special cases ----------------------------------------------------
    if p_hat <= 0.0:  # p̂ == 0
        return 1.0 - np.exp(-eps)
    if p_hat >= 1.0:  # p̂ == 1
        return 1.0

    # Bisection on [p_hat, 1] -----------------------------------------
    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl(p_hat, mid) > eps:
            hi = mid  # too high -> go left
        else:
            lo = mid  # too low  -> go right
        if hi - lo < tol:
            break
    return hi  # hi is the first point with KL > eps


# ---------------------------------------------------------------------
# 3. Produce the plot
# ---------------------------------------------------------------------
n = 1000
delta = 0.01

p_hats = np.linspace(0.0, 1.0, 1001)  # 0, 0.001, … ,1
bounds = np.array([kl_upper_bound(ph, n, delta) for ph in p_hats])


# ----
# Hoeffding error term  \epsilon = sqrt( ln(1/δ) / (2n) )
epsilon = np.sqrt(np.log(1.0 / delta) / (2.0 * n))

# Upper‐bound on the true bias  p  (may exceed 1 slightly; clip at 1 for display)
upper_bound = np.clip(p_hats + epsilon, 0.0, 1.0)
# ---

plt.figure(figsize=(6, 4))
plt.plot(p_hats, bounds, label="KL upper bound $p^+$")
plt.plot(p_hats, upper_bound, label="Hoeffding upper bound $p$")
plt.plot(p_hats, p_hats, "--", color="gray", label=r"$p = \hat p$")
plt.xlabel(r"Empirical mean $\hat p$")
plt.ylabel(r"Upper bound on $p$")
plt.title(
    r"One–sided $(1-\delta)$ KL–confidence bound"
    f"\n n = {n},  $\delta$ = {delta}"
)
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
