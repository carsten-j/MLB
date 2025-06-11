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


# ------------------------------------------------------------------
# Experiment set-up
# ------------------------------------------------------------------
n = 100
delta = 0.05
rng = np.random.default_rng(0)
K = 2  # two binary components in split-KL
delta_s = delta / (n + 1)  # for the ordinary KL gap
delta_k = delta / K  # for the split-KL components

p_a_grid = np.linspace(0.0, 1.0, 201)
gap_KL = []  # ordinary KL gap  (global)
gap_splitKL = []  # new split-KL gap
real_error = []  # realised error  p - p̂  (for intuition)

for p_a in p_a_grid:
    # distribution  P(X=0)=P(X=1)= (1-p_a)/2 ,  P(X=0.5)=p_a
    p0 = p1 = (1.0 - p_a) / 2.0
    probs = [p0, p_a, p1]
    support = np.array([0.0, 0.5, 1.0])

    sample = rng.choice(support, size=n, p=probs)
    p_hat = sample.mean()

    # --------------------------------------------------------------
    # 1. ordinary KL gap    (epsilon = ln((n+1)/δ) / n )
    # --------------------------------------------------------------
    p_plus = kl_upper_bound(p_hat, n, delta_s)
    gap_KL.append(p_plus - p_hat)

    # --------------------------------------------------------------
    # 2. split-KL gap
    # --------------------------------------------------------------
    # binary sub-samples
    X1 = (sample >= 0.5).astype(int)
    X2 = (sample == 1.0).astype(int)
    p_hat1 = X1.mean()
    p_hat2 = X2.mean()

    p_plus1 = kl_upper_bound(p_hat1, n, delta_k)
    p_plus2 = kl_upper_bound(p_hat2, n, delta_k)

    split_gap = 0.5 * p_plus1 + 0.5 * p_plus2 - p_hat
    gap_splitKL.append(split_gap)

    # realised deviation of this sample from the truth (p = 0.5)
    real_error.append(0.5 - p_hat)

gap_KL = np.array(gap_KL)
gap_splitKL = np.array(gap_splitKL)
real_error = np.array(real_error)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(16, 9))
plt.plot(p_a_grid, gap_KL, label="ordinary KL gap", lw=2)
plt.plot(p_a_grid, gap_splitKL, label="split-KL gap", lw=2)
plt.plot(p_a_grid, real_error, "--", label="realised error (single sample)", lw=1)

plt.xlabel(r"$p_{1/2} \; (=P(X=0.5))$")
plt.ylabel(r"upper bound on $p-\hat p$")
plt.title(rf"KL vs. split-KL bounds  ($n={n}$, $\delta={delta}$)")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()

plt.savefig("exercise2.6_v2.png", dpi=600)
plt.show()
