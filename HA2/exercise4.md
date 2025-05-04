
## o3

Question 7 – Equivalence (10 pts)
Drop the non–negativity requirement λ ≥ 0 for the moment.

Build a new (linear) feature map
Φ : ℝᵐ → ℝⁿ ,  Φ(x) := (y₁⟨x₁,x⟩, y₂⟨x₂,x⟩,…,yₙ⟨xₙ,x⟩)ᵀ.

In matrix form, using the Gram matrix K with K_{ij}=x_i^{⊤}x_j,
Φ(x_i) = (y₁K_{1i}, … , y_nK_{ni})ᵀ = YK_{·,i},   where Y=diag(y).

Keep λ∈ℝⁿ as the parameter vector and keep the same bias b.
Then for every training example (x_i , y_i)

λ^{⊤}Φ(x_i) = Σ_{j=1}^{n} λ_j y_j x_j^{⊤}x_i.

Substitute this into the Sparse–SVM constraints (7):

1 − y_i(λ^{⊤}Φ(x_i)+b) ≤ 0  ∀i,

exactly the margin constraints of a linear SVM that uses the data
{ (Φ(x_i), y_i) }_{i=1}^{n}         (see eqs. 9–10).

The objective (6) is ½‖λ‖², which is the standard ½‖w‖² once we call
the weight vector w:=λ in the new feature space.

Hence, disregarding λ_i ≥ 0, Sparse-SVM is nothing but an ordinary hard–margin linear SVM applied to the transformed points Φ(x_i).
(Representer theorems already tell us that the optimum of the ordinary SVM lives in the span of the training points, so no expressive power is lost.)

Question 8 – Lagrange dual of Sparse-SVM (15 pts)
Now derive the dual including the bound λ ≥ 0.

Primal problem (rewrite g_i(λ,b) ≤ 0 and h_i(λ) ≤ 0)

minimise    f(λ)=½ λᵀλ
s.t.         g_i(λ,b)=1−y_i(Σ_j λ_j y_j K_{ij}+b) ≤0,     i=1,…,n
h_i(λ)=−λ_i ≤0,                                i=1,…,n

Introduce Lagrange multipliers α_i ≥ 0 (for g_i) and μ_i ≥ 0 (for h_i).
Let Y=diag(y), 1 the all-ones vector, and γ = Yα (element–wise γ_i = α_i y_i).

Lagrangian
L(λ,b,α,μ)=½ λᵀλ + αᵀ1 − γᵀK Y λ − b αᵀy − μᵀλ.

Stationarity

1. ∂L/∂b = −αᵀy = 0   ⇒   αᵀy = 0.                               (1)

2. ∂L/∂λ = λ − YKYα − μ = 0   ⇒   λ = YKYα + μ.            (2)

Insert (2) in L to eliminate λ and b:

 λᵀλ = (YKYα + μ)ᵀ(YKYα + μ)
        = αᵀY K² Y α + 2 μᵀYKY α + μᵀμ.

 γᵀK Y λ = αᵀY K² Y α + αᵀY K Y μ.

 μᵀλ = μᵀYKY α + μᵀμ.

Putting everything together and canceling the cross terms, the dual function becomes

g(α,μ) = αᵀ1 − ½ αᵀY K² Y α − ½ μᵀμ.

Maximisation over μ ≥ 0: because the last term is –½ μᵀμ, the
optimum is attained at μ*=0.
Therefore the inequality λ ≥ 0 is not active at the optimum (it will be
guaranteed later by complementary slackness).

Final dual

maximise   W(α)= Σ_{i=1}^{n} α_i − ½ Σ_{i,j=1}^{n} α_i α_j y_i y_j (K²){ij}
subject to  Σ{i=1}^{n} α_i y_i = 0,   α_i ≥ 0 ∀i.

Here K² = K K is the squared Gram matrix, i.e.
(K²){ij} = Σ{k=1}^{n} (x_i^{⊤}x_k)(x_k^{⊤}x_j).

Thus Sparse-SVM’s dual looks exactly like the usual hard–margin SVM dual,
except that the ordinary kernel K(x_i,x_j)=x_i^{⊤}x_j is replaced by the
new positive–semidefinite kernel

  K̃(x_i,x_j) = (K²)_{ij} = Σ_k ⟨x_i,x_k⟩⟨x_k,x_j⟩.
References
• C. Cortes & V. Vapnik, “Support‐Vector Networks”, Machine Learning 1995.
• Representer theorem: G. Wahba, “Spline Models for Observational Data”, 1990.

## Deepseek
**Question 7**  
To show that the Sparse SVM coincides with the original Linear SVM (ignoring \(\lambda_i \geq 0\)):  
- The original SVM minimizes \(\frac{1}{2}\|v\|^2\) with constraints \(y_i(v^\top x_i + b) \geq 1\).  
- In the Sparse SVM, \(w = \sum \lambda_i y_i x_i\), and the objective is \(\frac{1}{2}\sum \lambda_i^2\).  
- If the data points \(\{x_i\}\) are **orthonormal** (\(x_i^\top x_j = \delta_{ij}\)), then \(\|w\|^2 = \sum \lambda_i^2\), making the objectives identical.  
- The constraints also match, as \(w^\top x_i = \sum \lambda_j y_j x_j^\top x_i\), which simplifies under orthonormality.  
Thus, under orthonormal data, the Sparse SVM coincides with the original SVM.  

**Answer to Question 7**  
\boxed{\text{When the data points } \{x_i\} \text{ are orthonormal, the Sparse SVM coincides with the original Linear SVM.}}  

---

**Question 8**  
To derive the Lagrange dual of the Sparse SVM:  
1. **Lagrangian**:  
\[
\mathcal{L} = \frac{1}{2}\sum \lambda_i^2 + \sum \alpha_i\left[1 - y_i\left(\sum \lambda_j y_j x_j^\top x_i + b\right)\right] - \sum \beta_i \lambda_i
\]  
where \(\alpha_i \geq 0\), \(\beta_i \geq 0\).  

2. **Stationarity conditions**:  
   - \(\frac{\partial \mathcal{L}}{\partial \lambda_k} = \lambda_k - \sum_i \alpha_i y_i y_k x_k^\top x_i - \beta_k = 0 \implies \lambda_k = \sum_i \alpha_i y_i y_k x_k^\top x_i + \beta_k\).  
   - \(\frac{\partial \mathcal{L}}{\partial b} = -\sum \alpha_i y_i = 0 \implies \sum \alpha_i y_i = 0\).  

3. **Substitute \(\lambda_k\) and simplify**:  
   - After substitution and algebraic manipulation, the dual objective becomes:  
\[
\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^\top x_j)^2
\]  
   - **Constraints**:  
     - \(\sum \alpha_i y_i = 0\),  
     - \(\alpha_i \geq 0\).  

**Answer to Question 8**  
\boxed{
\begin{aligned}
&\text{Maximize}_{\alpha} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j (x_i^\top x_j)^2 \\
&\text{Subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0 \quad \forall i.
\end{aligned}
}


## Claude
Actually, I think I've been over-complicating this. Let me re-read the problem statement more carefully.
It says: "Ignoring the additional constraint: λᵢ ≥ 0 for all i, show that (Sparse SVM) coincides with an instance of the original (Linear SVM) optimization problem."
The key word here is "an instance." I don't have to show that they are equivalent for all possible datasets, just that the Sparse SVM is an instance (i.e., a specific case) of the Linear SVM.
Given that v = Σᵢ₌₁ⁿ λᵢyᵢxᵢ, we have the constraints of the two problems aligned. The only difference is in the objective function:
Linear SVM: (1/2)||v||² = (1/2)||Σᵢ₌₁ⁿ λᵢyᵢxᵢ||² = (1/2)Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ λᵢλⱼyᵢyⱼxᵢᵀxⱼ
Sparse SVM: (1/2)Σᵢ₌₁ⁿ λᵢ²
Now, if we have an instance where xᵢᵀxⱼ = 1 if i = j and xᵢᵀxⱼ = 0 if i ≠ j (i.e., orthogonal and unit-norm data points), then:
||v||² = Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ λᵢλⱼyᵢyⱼxᵢᵀxⱼ = Σᵢ₌₁ⁿ λᵢ²yᵢ² = Σᵢ₌₁ⁿ λᵢ²
So, in this instance, the Linear SVM and Sparse SVM objectives are equivalent. And since the constraints are already aligned, the two problems coincide for this instance.
Therefore, ignoring the λᵢ ≥ 0 constraint, the Sparse SVM coincides with an instance of the Linear SVM where the data points are orthogonal and have unit norm.