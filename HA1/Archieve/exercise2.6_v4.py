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


# --------------------------------------------------------------------
# 2.  Experiment set-up
# --------------------------------------------------------------------
n = 100
delta = 0.05
rng = np.random.default_rng(0)

p_a_grid = np.linspace(0.0, 1.0, 201)  # values for  P(X=0.5)

kl_gap_list = []  # ordinary KL gap  p⁺ − p̂
split_gap_list = []  # split-KL gap
real_error_list = []  # realised 0.5 − p̂   (true p = 0.5)

# --------------------------------------------------------------------
# 3.  Loop over the grid
# --------------------------------------------------------------------
for p_a in p_a_grid:
    p0 = p1 = (1 - p_a) / 2.0  # probabilities of 0 and 1
    probs = [p0, p_a, p1]
    support = np.array([0.0, 0.5, 1.0])

    X = rng.choice(support, size=n, p=probs)
    p_hat = X.mean()

    # ---------- ordinary KL bound  (ε = ln((n+1)/δ)/n) ---------------
    delta_star = delta / (n + 1)
    p_plus = kl_upper_bound(p_hat, n, delta_star)
    kl_gap = p_plus - p_hat
    kl_gap_list.append(kl_gap)

    # ---------- split-KL bound ---------------------------------------
    # binary segments: 𝟙{X ≥ 0.5}  and  𝟙{X ≥ 1}
    X_seg1 = (X >= 0.5).astype(float)
    X_seg2 = (X >= 1.0).astype(float)
    p_hat1 = X_seg1.mean()
    p_hat2 = X_seg2.mean()

    # ε = ln(K/δ)/n  with K = 2  ⇒  call kl_upper_bound with δ/2
    p_plus1 = kl_upper_bound(p_hat1, n, delta / 2)
    p_plus2 = kl_upper_bound(p_hat2, n, delta / 2)

    split_gap = 0.5 * (p_plus1 - p_hat1) + 0.5 * (p_plus2 - p_hat2)
    split_gap_list.append(split_gap)

    # realised error (for illustration only)
    real_error_list.append(0.5 - p_hat)

kl_gap_list = np.array(kl_gap_list)
split_gap_list = np.array(split_gap_list)
real_error_list = np.array(real_error_list)

# --------------------------------------------------------------------
# 4.  Plot
# --------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(p_a_grid, kl_gap_list, label="ordinary KL bound")
plt.plot(p_a_grid, split_gap_list, label="split-KL bound", ls="--")
plt.plot(p_a_grid, real_error_list, ":", label="realised error (one sample)")
plt.xlabel(r"$p_a = P(X=0.5)$")
plt.ylabel(r"upper bound on  $p-\hat p$")
plt.title(rf"KL vs. split-KL   ($n={n}$,  $\delta={delta}$)")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
