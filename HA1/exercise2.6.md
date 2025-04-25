USER:
For the following we still need to invert KL using same process as the one we already used.
Take a ternary random variable (a random variable taking three values)
X ∈ {0, 0.5, 1}

Let p0 = Pr(X = 0), p_a = Pr(X = 0.5) and p1 = Pr(X = 1)

Set p0 = p1 = (1-p_a) / 2

i.e. the probabilities of X = 0 and X = 1 are equal,
and there is just one parameter p_a, which controls the probability mass of
the central value.

Let p = E[X]

For each value of p_a in a grid covering the interval [0, 1]
draw a random sample X1, . . . , Xn from the distribution we have constructed
and let p^n =  p^ = 1/n \sum_{i=1}^n  Xi.

Generate a figure in which you plot the kl bounds on p − pˆ
as a function of p_a for p_a ∈ [0, 1].

For the kl bound, the bound on p − pˆ is kl^{−1+}(\hat p, \epsilon) - p^ where

kl^{−1+}(\hat p, \epsilon) := max { p : p \in [0, 1]  and  kl(\hat p | p) < \epsilon }

with \epsilon = ln((n + 1)/delta) / n

Take n = 100 and delta = 0.05


ASSISTENT:

Reasoning:
Analyzing KL bounds plot

I need to generate a plot of KL bounds on p - p_hat as a function of p_a, with p_a values ranging from 0 to 1. I'll use a grid of 201 points for this.

For each p_a, I define the distribution where p0 and p1 are both (1 - p_a) / 2. When calculating the expectation of p, I find it interesting that p ends up being consistently 0.5, regardless of the p_a value. This constancy is quite intriguing!

Verifying expectation calculations

I'm verifying the symmetric distribution around 0.5, where p0 is 0, p_a is 0.5, and p1 is 1. The probabilities are p0 = p1 = (1 - p_a)/2. When calculating the expectation, I see that I can express it as E[X] = 0 + 0.5 * (p_a) + 1 * (1 - p_a)/2, which simplifies to 0.5, confirming that the true mean is constant at 0.5 for any p_a. This will help when plotting bounds on p - p_hat.

Next, for each p_a, I’ll draw random samples to compute the sample mean p_hat and calculate epsilon. It looks like I’ll compute the upper KL inverse bound for the Bernoulli distribution, even though the distribution has become ternary. I'll treat p_hat as the sample mean from {0, 0.5, 1}.

Bounding p using KL inequality

I need to bound p using the KL inequality for variables within [0,1]. The approach involves using the KL divergence for Bernoulli distributions, especially when comparing parameter p to p_hat.

I'll compute the sample mean for i.i.d. variables in this range and establish the necessary inequalities. We will generate a grid of p_a values, calculate the upper KL bounds, and plot both the bound differences and the actual differences across p_a. It’s crucial to define delta_new for accurate calculations, keep a random seed for reproducibility, and craft the necessary code for the plot!
-- end of Reasoning

answers:
Below we reuse the Bernoulli–KL inversion routine in a new experiment
with a ternary variable

    X ∈ {0 , 0.5 , 1}
    P(X = 0)     = P(X = 1) = (1 – p_a)/2
    P(X = 0.5)   = p_a ,     p_a ∈ [0,1].
For every p_a on a grid we

draw a sample of size n = 100,
compute the empirical mean p̂,
invert the KL–inequality to get the upper bound p⁺ = KL⁻¹₊(p̂, ε) , with ε = ln((n+1)/δ) / n ( δ = 0.05 ),
store the gap p⁺ − p̂ and finally plot it against p_a.
Remember:  in the function kl_upper_bound the parameter ε is internally
ln(1/δ*) / n;  hence we simply pass

    δ* = δ / (n+1)
so that ε matches the required expression.

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1.  KL divergence and its upper inverse for Bernoulli RVs
# --------------------------------------------------------------------
def kl(p_hat, p):
    """KL( Ber(p_hat) || Ber(p) ), safe at the boundaries."""
    p_hat = np.clip(p_hat, 1e-12, 1-1e-12)
    p     = np.clip(p,     1e-12, 1-1e-12)
    return p_hat*np.log(p_hat/p) + (1-p_hat)*np.log((1-p_hat)/(1-p))

def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    """Upper inverse of the KL-divergence via bisection."""
    eps = np.log(1.0/delta) / n

    # special cases
    if p_hat <= 0.0:
        return 1.0 - np.exp(-eps)
    if p_hat >= 1.0:
        return 1.0

    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        if kl(p_hat, mid) > eps:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

# --------------------------------------------------------------------
# 2.  Simulation parameters
# --------------------------------------------------------------------
n      = 100
delta  = 0.05
delta_star = delta / (n + 1)          # makes  ε = ln((n+1)/δ)/n
rng    = np.random.default_rng(0)

p_a_grid = np.linspace(0.0, 1.0, 201)   # 0, 0.005, … ,1
gap_vals = []                           # p⁺ − p̂ for each p_a
real_dev = []                           # actual deviation 0.5 − p̂ (for curiosity)

# --------------------------------------------------------------------
# 3.  Experiment on the grid
# --------------------------------------------------------------------
for p_a in p_a_grid:
    p0 = p1 = (1.0 - p_a) / 2.0
    probs = [p0, p_a, p1]
    support = np.array([0.0, 0.5, 1.0])

    sample = rng.choice(support, size=n, p=probs)
    p_hat  = sample.mean()

    p_plus = kl_upper_bound(p_hat, n, delta_star)
    gap_vals.append(p_plus - p_hat)
    real_dev.append(0.5 - p_hat)       # p = 0.5 for all p_a

gap_vals  = np.array(gap_vals)
real_dev  = np.array(real_dev)

# --------------------------------------------------------------------
# 4.  Plot  gap  versus p_a
# --------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(p_a_grid, gap_vals, label=r'KL bound  $p^{+}-\hat p$')
plt.plot(p_a_grid, real_dev, '--', label=r'Actual error  $p - \hat p$')

plt.xlabel(r'$p_a \; (=P(X=0.5))$')
plt.ylabel(r'Upper bound / realised error')
plt.title(fr'KL upper bounds on $p-\hat p$   ( $n={n}$ ,  $\delta={delta}$ )')
plt.legend()
plt.grid(True, ls=':')
plt.tight_layout()
plt.show()
What the figure shows
• Because the true mean is fixed at 0.5 for every p_a, the empirical mean
fluctuates around 0.5, with variance depending on the distribution’s
shape.
• The solid curve is the high-probability guarantee
p − p̂ ≤ p⁺ − p̂,
obtained by inverting the KL concentration inequality exactly as in
the Bernoulli case.
• The dashed curve is the realised error in the single sample that was
drawn for each p_a (included only for intuition; it is not guaranteed
to lie below the bound everywhere, but it usually does).

You can increase n, change the random seed, or repeat the simulation
to see how the bound tightens and how frequently it actually covers the
true parameter 0.5.





USER:
Great. Let us continue this example and enhance it with the concept of split-KL. Please find attached an image explaining split-KL. For the split-kl bound you subtract ^ pn from the right hand side of the expression inside the probability in Theorem 2.34. Please compute the upper bound for split-KL and add this to the visualization such that one can compare KL and spiit-KL


ASSISTENT:

Reasoning:
I'm working through an expectation problem involving probabilities. I lay out my equation, checking how p equals 0.5 once I simplify. I realize E[X] is constant, which is interesting! The difference I'm looking at is between p and p_hat, ending up with 0.5 - p_hat. Next, I need to simulate a sample and be clear in my explanation about the use of delta. I make sure to apply the theorem correctly and prepare the final code snippet and explanation.
-- end of Reasoning

answers:

Split–KL for the ternary example
For a discrete r.v.
  X ∈ {b₀,…,b_K} with b₀<b₁<⋯<b_K,
Wu et al. (2024) decompose X into K binary “segments”

 X = b₀ + Σ_{j=1}^{K} α_j X_{|j},  
  X_{|j}=𝟙{X ≥ b_j},  α_j = b_j−b_{j−1}.                     (1)

Let

  p_{|j}=E[X_{|j}],     
  p̂_{|j}= (1/n) Σ_i X_{i|j}.

The split-KL inequality (Thm. 2.34) tells us that for every δ∈(0,1)

 P { p ≥ b₀ + Σ_j α_j kl⁻¹₊(p̂_{|j}, ε) } ≤ δ,  
  ε = (1/n) ln(K/δ),                                       (2)

where kl⁻¹₊ is the same Bernoulli upper inverse we already coded.

Subtracting the empirical mean
  p̂ = b₀ + Σ_j α_j p̂_{|j},
gives a high–probability bound on the deviation

 p − p̂ ≤ Σ_j α_j [ kl⁻¹₊(p̂_{|j}, ε) − p̂_{|j} ].          (3)

For our ternary variable

 support   {0, 0.5, 1}  ⇒  b₀=0, b₁=0.5, b₂=1,
 segments  K = 2,  α₁ = α₂ = 0.5.                         (4)

Hence (3) becomes

 gap_split := ½[ kl⁻¹₊(p̂_{|1}, ε) − p̂_{|1} ]
+ ½[ kl⁻¹₊(p̂_{|2}, ε) − p̂_{|2} ],            (5)

with
 ε = (1/n) ln( K/δ ) = (1/n) ln(2/δ).

Because kl_upper_bound(p̂, n, δ*) uses ε = ln(1/δ*)/n, simply call it
with δ* = δ/K = δ/2.

Below the two bounds (ordinary KL and split-KL) are compared on exactly
the same simulated samples that we used before.

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1.   KL divergence and its upper inverse (already implemented)
# --------------------------------------------------------------------
def kl(p_hat, p):
    p_hat = np.clip(p_hat, 1e-12, 1-1e-12)
    p     = np.clip(p,     1e-12, 1-1e-12)
    return p_hat*np.log(p_hat/p) + (1-p_hat)*np.log((1-p_hat)/(1-p))

def kl_upper_bound(p_hat, n, delta, tol=1e-12, max_iter=60):
    eps = np.log(1.0/delta) / n
    if p_hat <= 0.0:
        return 1.0 - np.exp(-eps)
    if p_hat >= 1.0:
        return 1.0
    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
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
n      = 100
delta  = 0.05
rng    = np.random.default_rng(0)

p_a_grid = np.linspace(0.0, 1.0, 201)        # values for  P(X=0.5)

kl_gap_list      = []    # ordinary KL gap  p⁺ − p̂
split_gap_list   = []    # split-KL gap
real_error_list  = []    # realised 0.5 − p̂   (true p = 0.5)

# --------------------------------------------------------------------
# 3.  Loop over the grid
# --------------------------------------------------------------------
for p_a in p_a_grid:
    p0 = p1 = (1 - p_a) / 2.0           # probabilities of 0 and 1
    probs    = [p0, p_a, p1]
    support  = np.array([0.0, 0.5, 1.0])

    X = rng.choice(support, size=n, p=probs)
    p_hat = X.mean()

    # ---------- ordinary KL bound  (ε = ln((n+1)/δ)/n) ---------------
    delta_star = delta / (n + 1)
    p_plus     = kl_upper_bound(p_hat, n, delta_star)
    kl_gap     = p_plus - p_hat
    kl_gap_list.append(kl_gap)

    # ---------- split-KL bound ---------------------------------------
    # binary segments: 𝟙{X ≥ 0.5}  and  𝟙{X ≥ 1}
    X_seg1 = (X >= 0.5).astype(float)
    X_seg2 = (X >= 1.0).astype(float)
    p_hat1 = X_seg1.mean()
    p_hat2 = X_seg2.mean()

    # ε = ln(K/δ)/n  with K = 2  ⇒  call kl_upper_bound with δ/2
    p_plus1 = kl_upper_bound(p_hat1, n, delta/2)
    p_plus2 = kl_upper_bound(p_hat2, n, delta/2)

    split_gap = 0.5*(p_plus1 - p_hat1) + 0.5*(p_plus2 - p_hat2)
    split_gap_list.append(split_gap)

    # realised error (for illustration only)
    real_error_list.append(0.5 - p_hat)

kl_gap_list    = np.array(kl_gap_list)
split_gap_list = np.array(split_gap_list)
real_error_list= np.array(real_error_list)

# --------------------------------------------------------------------
# 4.  Plot
# --------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot(p_a_grid, kl_gap_list,      label='ordinary KL bound')
plt.plot(p_a_grid, split_gap_list,   label='split-KL bound', ls='--')
plt.plot(p_a_grid, real_error_list,  ':', label='realised error (one sample)')
plt.xlabel(r'$p_a = P(X=0.5)$')
plt.ylabel(r'upper bound on  $p-\hat p$')
plt.title(fr'KL vs. split-KL   ($n={n}$,  $\delta={delta}$)')
plt.legend()
plt.grid(True, ls=':')
plt.tight_layout()
plt.show()

What the plot shows
ordinary KL (solid) treats the ternary variable as if it were Bernoulli on the average value alone.
split-KL (dashed) exploits the two-segment decomposition (0.5 ↔ 1) and therefore uses two Bernoulli inversions with a milder penalty ln(2/δ) instead of ln((n+1)/δ). Consequently its curve lies below the ordinary KL bound, sometimes substantially so.
The dotted curve is the realised error for the single set of samples
that were drawn; it is not guaranteed to stay below either bound on
every point (because we are plotting just one run), but with
probability ≥ 1 − δ both confidence bounds cover the true mean 0.5
simultaneously.

USER:
I understand why the split-KL is below KL but can you explain why the difference between split-KL and KL increases as p_a get closer and closer to 1

Reasoning:

Explaining variance in p-hat

I'm analyzing the empirical mean p-hat, which is the average of sample X_i. The variation of p-hat depends on the variance of X, which can be calculated as Var(X) = E[X^2] - E[X]^2. I find that E[X^2] equals 0.5 - 0.25p_a, leading to variance shrinking linearly with (1 - p_a). When p_a approaches 1, the variance goes to 0, making the sample mean close to 0.5. This ties into the split-KL bound, leveraging reduced variance with segment decomposition.

Analyzing KL bounds

For X_{|1}, when p̂_{|1} is close to 1, the KL inverse with a small epsilon results in p_plus being close to 1, with a small difference from p_hat. For X_{|2}, with a small p_hat2 and weight 0.5, the difference might not vanish. As p_a approaches 1, the gap diminishes, but may remain moderate. The ordinary KL bound uses a Bernoulli distribution, constant across p_a. Our simulations show the KL gap might grow larger when p_a is near 0 or 1, and the split-KL gap diminishes with lower variance.

Explaining split-KL bounds

When p_hat is around 0.5, the gap is approximately constant and independent of p_a. For the split-KL gap, it accounts for differences in segments. As p_a approaches 1, p_hat1 gets close to 1 and the difference shrinks because the KL value approaches the boundary. With a Bernoulli variable, the gap becomes exponential in epsilon, leading to smaller differences due to factors like (1-p_hat). The split-KL method recognizes variance reduction, tightening its bound, while ordinary KL does not account for segment heterogeneity, allowing the gap to grow.

-- end of Reasoning

answers:

Why does the split-KL curve dive down while the “ordinary” KL curve
stays almost horizontal when p_a gets close to 1?

What changes when p_a→1
• Mass is pushed to the middle value 0.5.
• The two extreme outcomes (0 and 1) become very rare
(their probabilities p0 = p1 = (1–p_a)/2 go to 0).
• Consequently the variance of X shrinks:
Var(X)=0.25(1−p_a) → 0.
Hence the empirical mean p̂ is in practice locked very tightly
around the true mean 0.5 once p_a is large.

How the ordinary KL bound reacts
The one–dimensional KL bound treats X as if it were Bernoulli on
{0,1} with the same mean.
That is the “worst-case” distribution for the mean only, so the
resulting confidence radius has to work even for a distribution that
can jump between the two extreme values with probability ½ each.
Because this fictitious distribution does not change with p_a,
the bound it induces is (almost) constant all along the axis.
In other words, ordinary KL ignores the variance reduction that
comes from piling mass onto the central value 0.5.

How the split-KL bound reacts
Split-KL decomposes X into the two binary indicators

 X_|1 = 𝟙{X≥0.5},  X_|2 = 𝟙{X≥1}.
Their expectations are

 p_|1 = P(X≥0.5) = 1 – p0  → 1,
 p_|2 = P(X≥1  ) = p1      → 0  as p_a→1.
• When a Bernoulli parameter is very close to 1 or 0, the
KL-inverse difference kl⁻¹₊(p̂,ε) − p̂ is proportional to
p̂(1−p̂) and therefore goes to zero.
• In the split-KL bound these two tiny differences are multiplied by
α₁ = α₂ = 0.5 and then added, so the whole bound collapses
toward 0 as soon as X almost never takes the extreme values.

Thus split-KL automatically exploits the fact that “nothing happens
on the edges” when p_a is high, while ordinary KL cannot.

Resulting gap between the two curves
• Ordinary KL ≈ constant (it protects against a distribution that
could still jump to 0 or 1 half of the time).
• Split-KL → 0 (because in reality those jumps almost never happen).

Therefore the difference “ordinary KL – split-KL’’
widens steadily as p_a approaches 1.

Mathematically, for a Bernoulli indicator with empirical mean q̂,

kl⁻¹₊(q̂, ε) − q̂  ≈ √(2ε q̂(1−q̂))            for small ε,
so the term is throttled by the factor q̂(1−q̂).
With p_a→1 we have q̂_|1→1 and q̂_|2→0, hence both factors vanish, and
so does the split-KL radius, whereas the one-dimensional KL radius
contains no such factor and therefore stays put.