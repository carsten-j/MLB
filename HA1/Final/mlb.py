import numpy as np


def kl(p_hat, p, tol=1e-12):
    """
    Calculate the Kullback-Leibler divergence between two Bernoulli
    distributions.

    This function computes
    KL(p_hat || p) = p_hat * log(p_hat/p) + (1-p_hat) * log((1-p_hat)/(1-p))
    for two probability values p_hat and p.

    Parameters
    ----------
    p_hat : float
        First probability value (often an empirical estimate) in [0, 1].
    p : float
        Second probability value in [0, 1].
    tol : float, optional
        Tolerance for clipping the input values to avoid numerical issues.
        Default is 1e-12.

    Returns
    -------
    float
        The KL divergence between Bernoulli distributions with parameters
        p_hat and p.

    Notes
    -----
    Input values are clipped to avoid numerical issues when p_hat or p is very
    close to 0 or 1, which would cause log(0) calculations.

    This function corresponds to definition 2.14 in the lecture notes
    "Machine Learning The Science of Selection under Uncertainty" by Yevgeny Seldin
    """
    p_hat = np.clip(p_hat, tol, 1.0 - tol)
    p = np.clip(p, tol, 1.0 - tol)
    return p_hat * np.log(p_hat / p) + (1.0 - p_hat) * np.log((1.0 - p_hat) / (1.0 - p))


def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=100):
    """
    Calculate the upper confidence bound using the Kullback-Leibler divergence.

    This function computes an upper bound p such that
    KL(p_hat, p) ≤ log(1/delta)/n, where p_hat is the empirical probability
    estimate, n is the sample size, and delta is the confidence level.

    Parameters
    ----------
    p_hat : float
        The empirical probability estimate in [0, 1].
    n : int
        Number of samples.
    delta : float
        Confidence level parameter in (0, 1).
        The bound holds with probability 1-delta.
    tol : float, optional
        Tolerance for the binary search convergence. Default is 1e-12.
    max_iter : int, optional
        Maximum number of iterations for the binary search. Default is 100.

    Returns
    -------
    float
        The upper confidence bound p such that KL(p_hat, p) ≤ log(1/delta)/n.

    Notes
    -----
    Uses binary search to find the upper bound. Special cases are handled for
    p_hat = 0 and p_hat = 1.
    """
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


def kl_lower_bound_symm(p_hat, n, delta, **kw):
    """
    Calculate the lower (1-δ)-confidence bound for a Bernoulli parameter p
    by exploiting the symmetry of the KL divergence.

    This function finds p_lower such that
        KL(p_hat ‖ p_lower) ≤ log(1/δ) / n
    using the identity
        KL(p_hat ‖ p) = KL(1 - p_hat ‖ 1 - p).
    Internally, it computes the upper bound on the complementary probability
    (1 - p) via kl_upper_bound and returns its complement.

    Parameters
    ----------
    p_hat : float
        Empirical estimate of the Bernoulli probability ∈ [0, 1].
    n : int
        Number of independent trials.
    delta : float
        Confidence parameter ∈ (0, 1). The guarantee is that
        the true p ≥ p_lower with probability ≥ 1 - δ.
    **kw : dict
        Additional keyword arguments passed to kl_upper_bound
        (e.g., tol, max_iter).

    Returns
    -------
    float
        Lower confidence bound p_lower such that
        Pr[p ≥ p_lower] ≥ 1 - δ.

    Notes
    -----
    - Special cases for p_hat = 0 and p_hat = 1 are handled by
      kl_upper_bound on the complementary probability.
    - See kl_upper_bound for details on the binary‐search algorithm.
    """
    return 1.0 - kl_upper_bound(1.0 - p_hat, n, delta, **kw)


# --------------------------------------------------------------------
# Lower inverse via direct bisection
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
# Test cases comparing both versions
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
