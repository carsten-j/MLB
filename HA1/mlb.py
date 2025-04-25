import numpy as np


def kl(p_hat, p):
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

    Returns
    -------
    float
        The KL divergence between Bernoulli distributions with parameters
        p_hat and p.

    Notes
    -----
    Input values are clipped to avoid numerical issues when p_hat or p is very
    close to 0 or 1, which would cause log(0) calculations.
    """
    p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))


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
