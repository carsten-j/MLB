import numpy as np


# --------------------------------------------------------------------
# 1.  KL divergence for Bernoulli variables
# --------------------------------------------------------------------
def kl(p_hat, p):
    """
    KL( Ber(p_hat) || Ber(p) )  with  0·log0 interpreted as 0.
    Arrays are accepted.
    """
    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))


# --------------------------------------------------------------------
# 2.  Upper inverse (binary search)
# --------------------------------------------------------------------
def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """
    Return p⁺ ≥ p_hat such that  KL(p_hat||p⁺)=ln(1/delta)/n.
    When p ≤ p⁺ the KL–inequality guarantees P(event) ≥ 1-δ.
    """
    eps = np.log(1.0 / delta) / n

    # --- special cases ------------------------------------------------
    if p_hat <= 0.0:  # empirical mean 0
        return 1.0 - np.exp(-eps)
    if p_hat >= 1.0:  # empirical mean 1
        return 1.0

    # --- bisection on [p_hat , 1] ------------------------------------
    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl(p_hat, mid) > eps:
            hi = mid  # mid is too far to the right
        else:
            lo = mid  # mid still satisfies inequality
        if hi - lo < tol:
            break
    return hi  # first point with KL > eps  (lo is last OK)


# --------------------------------------------------------------------
# 3a.  Lower inverse via symmetry
# --------------------------------------------------------------------
def kl_lower_bound_symm(p_hat, n, delta, **kw):
    """
    Lower (1-δ)-confidence bound for p using the identity
         KL(p̂‖p) = KL(1-p̂ ‖ 1-p).
    """
    return 1.0 - kl_upper_bound(1.0 - p_hat, n, delta, **kw)


# --------------------------------------------------------------------
# 3b.  Lower inverse via direct bisection
# --------------------------------------------------------------------
def kl_lower_bound_bisect(p_hat, n, delta, tol=1e-12, max_iter=60):
    eps = np.log(1.0 / delta) / n

    # --- special cases -----------------------------------------------
    if p_hat <= 0.0:  # empirical mean 0
        return 0.0
    if p_hat >= 1.0:  # empirical mean 1
        return np.exp(-eps)

    # --- bisection on [0 , p_hat] ------------------------------------
    lo, hi = 0.0, p_hat
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl(p_hat, mid) > eps:
            lo = mid  # mid is too far to the left (KL too big)
        else:
            hi = mid  # mid still satisfies inequality
        if hi - lo < tol:
            break
    return hi  # first point with KL > eps (lo last OK)


# --------------------------------------------------------------------
# 4.  Test cases comparing both versions
# --------------------------------------------------------------------
def test_lower_bounds():
    rng = np.random.default_rng(1)
    n = 1234
    delta = 0.03
    atol = 1e-10  # tolerance for equality check

    # Edge cases ------------------------------------------------------
    edge_ps = np.array([0.0, 1.0, 1e-12, 1 - 1e-12])
    for p_hat in edge_ps:
        lb1 = kl_lower_bound_symm(p_hat, n, delta)
        lb2 = kl_lower_bound_bisect(p_hat, n, delta)
        assert np.allclose(lb1, lb2, atol=atol)

    # Random interior points -----------------------------------------
    p_hats = rng.uniform(0, 1, size=200)
    for p_hat in p_hats:
        lb1 = kl_lower_bound_symm(p_hat, n, delta)
        lb2 = kl_lower_bound_bisect(p_hat, n, delta)
        assert np.allclose(lb1, lb2, atol=atol)

    print(
        "All lower–bound tests passed ({} points).".format(len(edge_ps) + len(p_hats))
    )


# run the test suite
if __name__ == "__main__":
    test_lower_bounds()
