import math

import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# 1.  helper:  KL( Bern(p̂) || Bern(p) )
# --------------------------------------------------------------------
def kl(p_hat, p):
    """Kullback–Leibler divergence between two Bernoulli laws."""
    if p_hat == 0.0:
        return math.log(1.0 / (1.0 - p))
    if p_hat == 1.0:
        return math.log(1.0 / p)
    return p_hat * math.log(p_hat / p) + (1.0 - p_hat) * math.log(
        (1.0 - p_hat) / (1.0 - p)
    )


# --------------------------------------------------------------------
# 2.  upper bound obtained by inverting the KL inequality
# --------------------------------------------------------------------
def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """
    Return U such that     KL(p̂ || U) = ln(1/δ)/n     with U ≥ p̂.
    >  With probability ≥ 1-δ the true Bernoulli mean p is ≤ U.
    """
    bound = math.log(1.0 / delta) / n

    # degenerate corners
    if p_hat == 0.0:
        return 1.0 - math.exp(-bound)
    if p_hat == 1.0:
        return 1.0

    # bisection on (p̂ , 1)
    lo, hi = p_hat, 1.0 - 1e-15
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if kl(p_hat, mid) > bound:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi  # any point in [lo,hi] is admissible


# --------------------------------------------------------------------
# 3.  evaluate and plot for n = 1000 ,  δ = 0.01
# --------------------------------------------------------------------
n = 1000
delta = 0.01

p_hats = [k / n for k in range(n + 1)]
uppers = [kl_upper_bound(ph, n, delta) for ph in p_hats]

plt.figure(figsize=(6, 4))
plt.plot(p_hats, uppers, lw=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)  # y = x for reference
plt.xlabel(r"Empirical mean $\hat p$")
plt.ylabel(r"Upper bound $U(\hat p)$")
plt.title(r"KL upper confidence bound   ($n=1000,\;\delta=0.01$)")
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
