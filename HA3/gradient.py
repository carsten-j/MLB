import numpy as np


def logistic_gradients(
    w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, mu: float = 0.0
):
    """
    Parameters
    ----------
    w  : (m,)  weight vector
    b  : float bias term
    X  : (n, m) data matrix (rows = samples)
    y  : (n,)  labels in {0,1}
    mu : float L2‐regularisation coefficient

    Returns
    -------
    grad_w : (m,) gradient of L w.r.t. w
    grad_b : float gradient of L w.r.t. b
    """
    n = X.shape[0]

    # forward pass
    z = X @ w + b  # (n,)
    p = 1.0 / (1.0 + np.exp(-z))  # σ(z)

    # common term  (p - y) = ∂ℓᵢ / ∂zᵢ
    err = p - y  # (n,)

    # gradients
    grad_w = (X.T @ err) / n + mu * w  # (m,)
    grad_b = err.sum() / n + mu * b  # scalar

    return grad_w, grad_b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m = 200, 5
    X = rng.normal(size=(n, m))
    y = rng.integers(0, 2, size=n)
    w = rng.normal(size=m)
    b = rng.normal()
    mu = 0.1

    g_w, g_b = logistic_gradients(w, b, X, y, mu)

    # finite-difference sanity check on a single component of w
    eps = 1e-5
    k = 0  # perturb weight 0
    w_plus = w.copy()
    w_plus[k] += eps
    w_minus = w.copy()
    w_minus[k] -= eps

    def loss(w_, b_):
        z = X @ w_ + b_
        p = 1.0 / (1.0 + np.exp(-z))
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ce.mean() + (mu / 2) * (np.linalg.norm(w_) ** 2 + b_**2)

    fd_grad_k = (loss(w_plus, b) - loss(w_minus, b)) / (2 * eps)
    print("analytic :", g_w[k])
    print("finite-diff :", fd_grad_k)
