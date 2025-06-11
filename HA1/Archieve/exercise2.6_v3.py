import matplotlib.pyplot as plt
import numpy as np


def kl(p_hat, p):
    p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))


def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    eps = np.log(1.0 / delta) / n
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


# ----------  Experiment parameters ----------------------------------
n = 100
delta = 0.05
K = 2  # number of indicators in split-KL
delta_star = delta / (n + 1)  # for the *ordinary* KL bound
delta_split = delta / K  # so that ln(1/delta_split)=ln(K/δ)

rng = np.random.default_rng(3317)
p_a_grid = np.linspace(0.0, 1.0, 201)  # e.g. 0.0, 0.005, ..., 1.0


# ----------  Compute KL bounds --------------------------------------

gap_kl = []  # ordinary KL   :  p⁺   − p̂
gap_splitkl = []  # split-KL      :  p_split⁺ − p̂
real_error = []  # realised error:  p     − p̂   (reference)

for p_a in p_a_grid:
    p0 = p1 = (1.0 - p_a) / 2.0
    probs = [p0, p_a, p1]
    support = np.array([0.0, 0.5, 1.0])

    sample = rng.choice(support, size=n, p=probs)
    p_hat = sample.mean()  # empirical mean of X
    p_true = 0.5  # E[X] = 0.5 for all p_a

    # ---------- ordinary KL bound on X directly ----------------------
    p_plus = kl_upper_bound(p_hat, n, delta_star)
    gap_kl.append(p_plus - p_hat)

    # ---------- split-KL bound ---------------------------------------
    # empirical means of the two indicators
    p1_hat = np.mean(sample >= 0.5)  # proportion >=0.5
    p2_hat = np.mean(sample >= 1.0)  # proportion  ==1

    p1_plus = kl_upper_bound(p1_hat, n, delta_split)
    p2_plus = kl_upper_bound(p2_hat, n, delta_split)

    p_plus_split = 0.5 * (p1_plus + p2_plus)  # b0=0, α1=α2=0.5
    gap_splitkl.append(p_plus_split - p_hat)

    # realised error (for visual reference only)
    real_error.append(p_true - p_hat)

gap_kl = np.array(gap_kl)
gap_splitkl = np.array(gap_splitkl)
real_error = np.array(real_error)

# ----------------------------  Plot ---------------------------------
plt.figure(figsize=(6.4, 4.2))
plt.plot(p_a_grid, gap_kl, label="ordinary KL")
plt.plot(p_a_grid, gap_splitkl, label="split-KL")
plt.plot(p_a_grid, real_error, "--", color="gray", label="realised error (one draw)")

plt.xlabel(r"$p_a \;(=P(X=0.5))$")
plt.ylabel(r"upper bound on $p - \hat p$")
plt.title(rf"KL vs. split-KL bounds   ($n={n}$, $\delta={delta}$)")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
