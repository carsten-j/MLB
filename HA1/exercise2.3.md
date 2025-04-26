User
we now use
kl^{−1-}(\hat p, \epsilon) := min { p : p \in [0, 1]  and kl(\hat p | p) < \epsilon }
to denote the lower inverse of kl.
Can you think of a clever way to compute this reusing the implementation of the upper inverse

Assistent:
Yes – one line is enough.

Key observation
For every a,b∈[0,1]

  KL(a‖b) = KL(1−a ‖ 1−b).                                (1)

Hence
 KL(p̂‖p) < ε is equivalent to KL(1−p̂ ‖ 1−p) < ε.

The set {p : KL(p̂‖p) < ε} is therefore obtained from the set
{q : KL(1−p̂‖q) < ε} by the change of variable q = 1−p.

Now recall the definition of the upper inverse

 q⁺ = KL⁻¹₊(1−p̂, ε) = max{q ≥ 1−p̂ : KL(1−p̂‖q) < ε}.

Replacing q by 1−p gives

 p⁻ = 1 − q⁺ = 1 − KL⁻¹₊(1−p̂, ε)
= min{p ≤ p̂ : KL(p̂‖p) < ε} = KL⁻¹₋(p̂, ε).       (2)

So the lower inverse is obtained from the upper inverse
of the complemented empirical mean.

Python implementation

def kl_lower_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """
    Lower (1–delta)-confidence bound for the Bernoulli mean p.

    Returns  p⁻  such that      P(p < p⁻) ≤ delta.
    Computed by the identity  p⁻ = 1 – KL⁻¹₊(1–p̂, ε).
    """
    return 1.0 - kl_upper_bound(1.0 - p_hat,
                                n=n,
                                delta=delta,
                                tol=tol,
                                max_iter=max_iter)
All special cases are handled automatically:

• p̂ = 0  ⇒ kl_upper_bound(1,·)=1 ⇒ p⁻ = 0
• p̂ = 1  ⇒ kl_upper_bound(0,·)=1−e^(−ε) ⇒ p⁻ = e^(−ε).

Thus the existing upper–bound routine is reused completely; no new
root-finder is required.