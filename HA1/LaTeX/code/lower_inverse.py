def kl_lower_bound_symm(p_hat, n, delta, **kw):
    return 1.0 - kl_upper_bound(1.0 - p_hat, n, delta, **kw)