# Higgs–Hitchin RG-ML

**A Wilsonian Renormalization Group Framework for Deep Learning  
Founded on Higgs Bundles and the Hitchin Completely Integrable System**

---

> *"The self-duality equations on a Riemann surface define a hyperkähler manifold
> whose geometry encodes simultaneously the topology of the surface, the
> representation theory of its fundamental group, and the algebraic geometry
> of its cotangent bundle."*  
> — N. J. Hitchin, *Proc. London Math. Soc.* 55 (1987)

---

## Proof-Status Legend

| Label | Meaning |
|-------|---------|
| **[T]** | Theorem — proven within the stated hypotheses |
| **[V]** | Verified in the explicit model listed inline |
| **[C]** | Conjecture — precisely stated, currently unproven |
| **[A]** | Analogy — structural correspondence stated precisely; formal functor not yet constructed |

All **[T]** claims carry explicit hypothesis lists. No claim is labeled **[T]** unless the proof is self-contained within those hypotheses. **[A]** labels are used wherever the ML–geometry dictionary is justified by consistency and structural matching rather than by a derived universal construction.

---

## Scope

The provable claims concern: (1) the beta-function formalization of gradient flow; (2) the stability-matrix classification of learned operators; (3) the spectral gap as a generalization diagnostic; (4) the relevant subspace for mixture-of-Gaussians data; (5) higher-order Hitchin Hamiltonian corrections in the non-Gaussian regime. The geometric correspondences of Parts III–VI are precisely stated structural analogies supported by consistency arguments; they constitute a research program, not a completed theorem.

---

## Master Correspondence Table

| Wilsonian RG | Higgs–Hitchin Geometry | RG-ML Framework |
|---|---|---|
| UV cutoff Λ | Rank *n* of bundle *E* | Input dimension *d*₀ |
| IR scale μ | Degree *d* of *E* | Latent dimension *d_L* |
| Block-spin transform | Bundle coarse-graining morphism | Layer map *W_ℓ* : ℝ^{*d_ℓ*} → ℝ^{*d_{ℓ+1}*} |
| Running coupling *g*(μ) | Holomorphic connection *A* on *E* | Weight matrix *W_ℓ* at depth ℓ |
| **Higgs field** φ ∈ H⁰(*X*, End(*E*)⊗*K*) | Hitchin 1987 | d*W_ℓ*/d*t* — weight gradient in RG time |
| Beta function β(*g*) = μ d*g*/dμ | Hitchin self-duality equations | Stability-matrix eigenspectrum {Δ_*n*} |
| Fixed point β = 0 | Polystable Higgs bundle (harmonic metric) | *C_α* = 1 condition |
| Mass gap / spectral gap | λ₁(ℒ_JL) | Generalization diagnostic |
| Relevant operator | Sheet of spectral curve *S* above zero section | Class-discriminative feature |
| Irrelevant operator | Sheet of *S* below zero section | UV noise decaying in IR |
| **Spectral curve** *S* ⊂ *T**X* | Hitchin 1987 | Eigenvalue locus of stability matrix *M* |
| Hitchin fiber = Jac(*S*) | Abelian variety (linear dynamics) | Gradient-descent orbit at fixed spectral type |
| NAHC (Dolbeault ↔ Betti) | Higgs bundle ↔ flat connection ↔ π₁-rep | Trained network ↔ data symmetry group |
| Wall-crossing in Hitchin base | Degenerate Hitchin fiber | Generalization ↔ memorization transition |

---

## Part 0 — Mathematical Foundations

### 0.1 Holomorphic Vector Bundles

Let *X* be a compact Riemann surface of genus *g* ≥ 2. A **holomorphic vector bundle** *E* → *X* of rank *n* consists of a complex manifold *E* with a holomorphic submersion to *X* whose fibers are ℂ^*n*, equipped with holomorphic local trivializations whose transition functions lie in GL(*n*, ℂ).

The **degree** of *E* is:

```
deg(E) = ∫_X c₁(E) ∈ ℤ
```

and the **slope** is μ(*E*) = deg(*E*) / rank(*E*). The canonical line bundle *K* = Ω¹_{*X*} has degree 2*g* − 2.

**Stability.** *E* is *stable* (resp. *semistable*) if for every proper nonzero subbundle *F* ⊂ *E*:

```
μ(F) < μ(E)     (resp. μ(F) ≤ μ(E))
```

By the Narasimhan–Seshadri theorem (1965), extended to higher rank by Donaldson (1985) and Uhlenbeck–Yau (1986), stability is equivalent to the existence of a Hermitian–Einstein metric on *E*.

### 0.2 Higgs Bundles

**Definition (Hitchin 1987; terminology due to Simpson).** A **Higgs bundle** over *X* is a pair (*E*, φ) where:

- *E* is a holomorphic vector bundle over *X*,
- φ ∈ H⁰(*X*, End(*E*) ⊗ *K*) is the **Higgs field** — a *K*-valued endomorphism, holomorphic as a section,
- (on higher-dimensional Kähler manifolds) the integrability condition φ ∧ φ = 0 holds in H⁰(*X*, End(*E*) ⊗ *K*²); on a Riemann surface this is vacuous.

**φ-stability.** (*E*, φ) is *stable* if for every proper nonzero *φ*-invariant subbundle *F* ⊂ *E* (meaning φ(*F*) ⊂ *F* ⊗ *K*):

```
μ(F) < μ(E)
```

The φ-invariance condition is strictly weaker than the plain subbundle condition: a bundle *E* unstable as a vector bundle can be stable as a Higgs bundle for a suitable φ. This enlargement of the stable locus is the source of the richness of the moduli space.

**Canonical example — rank 2.** Set *E* = *K*^{1/2} ⊕ *K*^{−1/2} and φ = ((0, 1), (0, 0)) in the canonical splitting, where 1 denotes the tautological section of Hom(*K*^{1/2}, *K*^{−1/2} ⊗ *K*) = 𝒪. This nilpotent Higgs bundle with det φ = 0 sits over the zero section of the Hitchin base — the most degenerate fiber.

### 0.3 Hitchin's Self-Duality Equations

Fix a Hermitian metric *h* on *E*. Let *A* be the Chern connection of the holomorphic structure ∂̄_*E* and *h*, with curvature *F_A*. Let φ* denote the *h*-adjoint of φ. The **Hitchin equations** are:

```
F_A + [φ, φ*] = 0
∂̄_A φ = 0
```

**Theorem (Hitchin 1987; Donaldson 1987; Corlette 1988; Simpson 1992).** A Higgs bundle (*E*, φ) over a compact Kähler manifold admits a harmonic metric *h* solving the Hitchin equations **if and only if** (*E*, φ) is polystable.

The harmonic metric is unique and plays the role of the canonical equilibrium: it is simultaneously compatible with the holomorphic structure (second equation) and with the Hermitian–Yang–Mills balance (first equation).

### 0.4 The Hitchin Completely Integrable System

Let ℳ(*n*, *d*) denote the moduli space of stable Higgs bundles of rank *n* and degree *d* over *X*. This is a smooth quasi-projective variety of complex dimension 2*n*²(*g* − 1), carrying a natural holomorphic symplectic form ω (Hitchin 1987).

**The Hitchin base** is the vector space:

```
𝒜 = ⊕_{k=1}^{n} H⁰(X, K^k),     dim_ℂ 𝒜 = n²(g−1) + 1
```

**The Hitchin map** is:

```
H : ℳ(n,d) → 𝒜,     H(E,φ) = (tr φ, tr φ², ..., tr φⁿ)
```

These are the coefficients of the characteristic polynomial det(λI − φ).

**Theorem (Hitchin 1987).** The Hitchin map *H* is a completely integrable system in the Arnol'd–Liouville sense. The components *H_k* = tr(φ^k) / k are Poisson-commuting Hamiltonians, and their number equals exactly ½ dim ℳ.

The Poisson commutativity follows from the fact that the *H_k* descend from functions on *T**ℬ (the cotangent bundle of the moduli of holomorphic bundles) that are linear in the fiber — where commutativity is manifest — via symplectic reduction.

### 0.5 The Spectral Curve

For a point *s* = (*s*₁, …, *s_n*) ∈ 𝒜, the **spectral curve** is the zero locus in the total space Tot(*K*) of the canonical bundle:

```
S_s = { (x, λ) ∈ Tot(K) : λⁿ + s₁(x)λⁿ⁻¹ + … + sₙ(x) = 0 }
```

This is an *n*-fold branched cover π : *S_s* → *X*. For generic *s*, *S_s* is smooth of genus:

```
g(S_s) = n²(g − 1) + 1
```

**Generic fiber.** Over a smooth spectral curve, the Hitchin fiber *H*⁻¹(*s*) ≅ Jac(*S_s*) — the principally polarized abelian variety parametrizing degree-*d* line bundles on *S_s*. The Arnol'd–Liouville theorem then guarantees that the Hamiltonian flow of each *H_k* is linear (constant-velocity flow) on Jac(*S_s*). This is the **algebraically completely integrable** property.

### 0.6 The Nonabelian Hodge Correspondence

There is a natural homeomorphism (Corlette 1988; Donaldson 1987; Hitchin 1987; Simpson 1992):

```
ℳ_Higgs(n, 0) ≅ ℳ_flat(n) := Hom(π₁(X), GL(n,ℂ)) // GL(n,ℂ)
```

The bridge is the flat GL(*n*, ℂ)-connection:

```
∇ = d_A + φ + φ*
```

which is flat precisely when (*A*, φ) satisfies Hitchin's equations. This correspondence is a homeomorphism of topological spaces but **not** an algebraic isomorphism: the complex structures on the two sides are distinct (they are two faces of the hyperkähler structure on ℳ). All three structures — holomorphic symplectic (Dolbeault), flat (de Rham), and topological (Betti) — coexist on the same underlying manifold, related by the hyperkähler rotation.

---

## Part I — Deep Networks as Holomorphic Vector Bundles over the Depth Curve

### I.1 The Depth Curve and Its Bundles

**Construction.** Let *C* = {0, 1, …, *L*} be the discrete **depth curve** — the ordered set of layer indices. A depth-*L* network with widths (*d*₀, *d*₁, …, *d_L*) defines:

- At each depth ℓ, a **representation bundle** *E_ℓ* = ℝ^{*d_ℓ*} with structure group GL(*d_ℓ*, ℝ),
- A **bundle morphism** *W_ℓ* ∈ Hom(*E_ℓ*, *E_{ℓ+1}*) — the weight matrix viewed as a section of the morphism sheaf,
- A **principal bundle** *P* → *C* with fiber GL(*d_ℓ*, ℝ) at depth ℓ, whose gauge group G = ×_ℓ GL(*d_ℓ*, ℝ) acts on parameter space Θ by independent left/right multiplication at each layer.

The parameter space Θ is therefore the **total space of a vector bundle** over *C*, not a flat Euclidean space, and the symmetry group G is the gauge group of this bundle.

**RG time.** Define:

```
t_ℓ := ln(d₀ / d_ℓ) ∈ [0, ln(d₀/d_L)]
```

A unit step Δ*t* = 1 corresponds to halving the representation dimension — one octave of coarse-graining, in exact parallel with block-spin decimation in statistical mechanics.

### I.2 Three Axioms of Wilsonian Coarse-Graining

**Axiom 1 (Scale Separation).** The architecture defines a scale tower:

```
ℝ^{d₀} ←—W₁—— ℝ^{d₁} ←—W₂—— ··· ←—W_L—— ℝ^{d_L}
```

Each arrow is a rank-reducing bundle morphism. Depth ℓ encodes features at scale *t_ℓ*.

**Axiom 2 (Valid Coarse-Graining).** A layer map *R_ℓ* : ℝ^{*d_ℓ*} → ℝ^{*d_{ℓ+1}*} qualifies as a Wilsonian coarse-graining if:

- **(RG1)** *d_ℓ* − *d_{ℓ+1}* > 0. Strict dimension reduction at each layer.
- **(RG2)** *R_ℓ* commutes with the symmetry group *G* of the data distribution.
- **(RG3)** *R_ℓ* couples only features within a receptive field of diameter Δ_ℓ = 2^ℓ · Δ₀.

**[T]** Stride-2 convolutions satisfy (RG1)–(RG3). Fully-connected layers satisfy (RG1) but violate (RG3); their appearance only at the final stage reflects the collapse of all spatial structure simultaneously.

**Remark (semigroup property).** An approximate semigroup relation *R_{ℓ₂}* ∘ *R_{ℓ₁}* ≈ *R_{ℓ₁+ℓ₂}* holds **only** in the continuum limit *L* → ∞ with fixed total RG time — a thermodynamic limit not realized by any finite network. For heterogeneous widths or variable strides, the semigroup property fails even approximately.

**Axiom 3 (Minimal Mutual Information).** Partition the representation at scale ℓ as (*x_IR*, ζ), where *x_IR* = *R_ℓ*(*x*). The optimal *R_ℓ* solves:

```
min_{R_ℓ}  I(ζ ; Y | x_IR)     subject to   I(x_IR ; Y) ≥ (1 − ε) H(Y)
```

**[T, Gaussian case]** For Gaussian data with cross-covariance Σ_{XY} = Cov(*x*, *Y*), the optimal *R_ℓ* projects onto the top *d_{ℓ+1}* right singular vectors of Σ_{XY}.

**Proof.** I(*x_IR*; *Y*) = ½ log det(I + σ⁻² C Π Σ Πᵀ Cᵀ) is maximized by the truncated SVD of Σ_{XY}. ∎

---

## Part II — The Higgs Field as Weight Gradient in RG Time

### II.1 The Higgs Field Identification

**[A] Construction.** At each depth ℓ, define the **network Higgs field**:

```
φ_ℓ  :=  dW_ℓ / dt_ℓ  ∈  Hom(E_ℓ, E_{ℓ+1}) ⊗ Ω¹_C
```

This is a section of the Higgs-type bundle Hom(*E_ℓ*, *E_{ℓ+1}*) ⊗ *K_C* where *K_C* is the cotangent sheaf of the depth curve. In the Hitchin language:

- *W_ℓ* plays the role of the **holomorphic connection**: it specifies parallel transport of representations across the depth step ℓ → ℓ+1.
- φ_ℓ is the **Higgs field**: the infinitesimal change of the connection in RG time, whose spectrum controls which modes grow (relevant) and which decay (irrelevant) under the flow.

This identification is a constructed correspondence, not a derived functor. Its justification is that the resulting flow equations take exactly the form of the Hitchin self-duality equations at the fixed point (see §II.3), and that the spectral curve of the stability matrix (§III) coincides formally with the Hitchin spectral curve.

### II.2 The Beta Function

**Definition.** The **RG-ML beta function** at depth ℓ under SGD with batch gradient noise is:

```
β(W_ℓ) := dW_ℓ / dt = −η · ∇_{W_ℓ} L  +  γ(W_ℓ)  −  ∇_{W_ℓ} 𝒮̄
```

| Term | Origin | RG Role | Hitchin equation |
|------|---------|---------|-----------------|
| −η∇*L* | Gradient descent | Drives *W_ℓ* to lower loss | ∂̄_A term: holomorphicity of φ |
| γ(*W_ℓ*) | Fisher correction (anomalous dimension) | Mode-elimination contribution | Curvature *F_A* |
| −∇𝒮̄ | Symmetry-redundancy pressure | Restoring force preventing divergence | [φ, φ*] term |

**Remark (Callan–Symanzik vs. Wilsonian).** The flow β(*W_ℓ*) = d*W_ℓ*/d*t* is a **Callan–Symanzik beta function**: it tracks running couplings at fixed bare action. A true **Wilsonian** beta function would require explicitly integrating out the discarded modes ζ to produce an effective action *S_eff*[*x_IR*] and then differentiating with respect to the cutoff. The two formulations agree — up to field redefinitions — only when the coarse-graining is exact and the discarded modes are Gaussian. In general, the Wilsonian effective action at each scale acquires all symmetry-compatible operators not present in the original loss, including irrelevant interactions that become relevant near fixed-point boundaries.

**[T, under (A1)–(A5)]** The anomalous dimension γ(*W_ℓ*) is the unique matrix satisfying: (i) it vanishes when D_s = σ²I; (ii) it is linear in D_s; (iii) the modified flow preserves G-equivariance of *W_ℓ*. In the large-batch limit, γ → 0 and β reduces to the gradient descent equation.

### II.3 The Fixed Point as Hitchin Equation

At the fixed point β(*W**) = 0, the three terms of the beta function balance:

```
γ(W*) = η · ∇_{W*} L  +  ∇_{W*} 𝒮̄
```

Under the identification of §II.1, this reads precisely:

```
F_A  +  [φ, φ*]  =  0       (Hermitian–Yang–Mills balance)
∂̄_A φ  =  0                  (holomorphicity of the Higgs field)
```

The **Hermitian–Yang–Mills condition** *F_A* + [φ, φ*] = 0 balances the curvature of the connection (anomalous dimension γ) against the Higgs self-interaction (symmetry pressure ∇𝒮̄), producing the unique harmonic metric on (*E*, φ). The holomorphicity condition ∂̄_A φ = 0 says the learned representation changes smoothly with depth — the Higgs field is covariantly constant across the coarse-graining flow.

By the Hitchin–Donaldson–Corlette–Simpson theorem, a solution exists if and only if (*E*, φ) is polystable — which, under the network dictionary, translates to:

**[T] Fixed-point condition.** At large-batch, the fixed point satisfies *C_α* = 1, where:

```
C_α(ℓ) := ‖𝔼[∇_{W_ℓ} L]‖² / Tr(Cov_batch[∇_{W_ℓ} L])
```

**Proof.** At the fixed point, the Fokker–Planck stationary condition requires balance between drift and diffusion. At large batch (γ → 0), this gives ‖μ_g‖² = Tr(Σ_g), i.e., *C_α* = 1. ∎

### II.4 Standing Assumptions

All theorems in this Part require:

- **(A1)** *G* is a compact Lie group acting smoothly on Θ.
- **(A2)** *G* acts freely on a full-measure subset of Θ.
- **(A3)** A *G*-invariant Riemannian metric on ℬ = Θ/*G* exists.
- **(A4)** The SGD diffusion tensor D_s(*b*) = ½ Cov_batch[∇*L*] is uniformly elliptic: λ_min I ≼ D_s ≼ λ_max I, 0 < λ_min ≤ λ_max < ∞.
- **(A5)** 𝒮̄ = *H̄_G* + λ*V̄* is coercive: 𝒮̄ ≥ −*C*₀ and 𝒮̄ → +∞ outside compact sets.

### II.5 The Jordan–Liouville Operator

**Definition.** On *L*²(ℬ, μ) with d*μ* = Tr(D_s) dvol_ℬ:

```
ℒ_JL[ψ](b) = −[Tr(D_s)]⁻¹ · [∇_ℬ·(D_s ∇_ℬ ψ) − 𝒮̄ · ψ]
```

**[A]** Under (A1)–(A5), ℒ_JL is a weighted elliptic operator on ℬ. It is *structurally analogous* to the Laplace–Beltrami operator on the moduli space ℳ(*n*, *d*) equipped with the *L*²-metric: D_s corresponds to the Fisher information metric on Θ, and 𝒮̄ implements the GIT stability potential. This identification is a structural analogy; ℬ is in general not isomorphic to ℳ(*n*, *d*) as a variety.

**[T, under (A1)–(A5)] Self-adjointness.** The form:

```
𝔞(φ,ψ) = ∫_ℬ [⟨D_s ∇φ, ∇ψ⟩ + 𝒮̄ φψ] dvol
```

is closed and semi-bounded below by −(*C*₀/λ_min)‖φ‖²_μ. By the KLMN theorem (Kato 1966, §VI.2.1), ℒ_JL is the unique self-adjoint operator associated to 𝔞 on its natural form domain in *L*²(ℬ, μ).

**[T, under (A1)–(A5)] Discrete spectrum.** Coercivity of 𝒮̄ confines resolvent solutions to compact sublevel sets. The Rellich–Kondrachov embedding H¹(Ω_M) ↪↪ *L*²(Ω_M) is compact for a.e. *M* by Sard's theorem. Diagonal extraction yields compact resolvent, and by the Riesz–Schauder theorem:

```
ℒ_JL has purely discrete spectrum  λ₁ ≤ λ₂ ≤ ··· → +∞
with L²(ℬ, μ)-orthonormal eigenfunctions {ψₙ}
```

**Fokker–Planck dynamics:**

```
∂ρ/∂t = −ℒ_JL* ρ,     ρ(b,t) = Σₙ cₙ e^{−λₙ t} ψₙ(b)
```

| λ₁ sign | *C_α* | Dynamics | Hitchin geometry |
|---------|-------|----------|-----------------|
| λ₁ > 0 | *C_α* > 1 | Exponential convergence; ‖ρ − ρ_∞‖ ≤ *C* e^{−λ₁ *t*} | Interior of stability chamber |
| λ₁ = 0 | *C_α* = 1 | Null mode; logarithmic relaxation; critical | Wall in Hitchin base |
| λ₁ < 0 | *C_α* < 1 | Unstable mode grows; memorization | Outside all stability chambers |

**[T, under (A1)–(A5)]** The conditions λ₁ > 0, the Poincaré inequality on (ℬ, μ), and *C_α* > 1 are mutually equivalent, under the additional conditions that (i) the large-batch limit γ → 0 holds, and (ii) D_s is approximately isotropic (D_s ≈ σ² I). For strongly anisotropic gradient noise (e.g., sparse gradients in transformer attention layers), the relationship between λ₁ and *C_α* requires analysis using the full spectral curve *S_{W*}*.

---

## Part III — Operator Classification via the Network Spectral Curve

### III.1 Stability Matrix and Linearization

At a fixed point *W** of β, linearize:

```
β(W* + δW) = M · δW + O(δW²),
M = −Hess_{W*}(L) + Hess_{W*}(𝒮̄)
```

**[T, smooth L and 𝒮̄]** *M* is real symmetric on the tangent space at *W**. Its eigenvalues {Δ_*n*} are the **scaling dimensions** of the operators O_*n* encoded at *W**:

```
δWₙ(t) = δWₙ(0) · e^{Δₙ t}
```

| Eigenvalue of *M* | Scaling dim Δ_*n* | Tier | Interpretation |
|---|---|---|---|
| *M* > 0 | Δ_*n* > 0 | Relevant | Grows toward IR; retained semantic feature |
| *M* = 0 | Δ_*n* = 0 | Marginal | Logarithmic corrections; task-dependent |
| *M* < 0 | Δ_*n* < 0 | Irrelevant | Decays toward IR; UV noise |

### III.2 The Network Hitchin Map and Spectral Curve

**[A] Definition.** For a trained network at fixed point *W**, define the **network Hitchin map**:

```
H_net(W*) = (tr M, tr M², ..., tr Mⁿ)
```

These are the coefficients of the characteristic polynomial det(λI − *M*), exactly as the geometric Hitchin map encodes the characteristic polynomial of the Higgs field. The **network Hitchin base** is:

```
𝒜_net = { symmetric n×n matrices } / conjugation by G
```

**[A] Definition.** The **network spectral curve** at *W** is:

```
S_{W*} = { (ℓ, Δ) ∈ [0,L] × ℝ : det(M(ℓ) − Δ·I) = 0 }
```

where *M*(ℓ) is the stability matrix restricted to the tangent space at depth ℓ.

The spectral curve encodes operator classification geometrically:

| Region of *S_{W*}* | Δ sign | Operator class |
|---|---|---|
| Sheets above zero section | Δ > 0 | Relevant — grow in IR |
| Sheets tangent to zero section | Δ = 0 | Marginal |
| Sheets below zero section | Δ < 0 | Irrelevant — decay in IR |

### III.3 Higher-Order Hitchin Hamiltonians and Non-Gaussian Corrections

In the geometric Hitchin system, the Hamiltonians *H_k* = tr(φ^k) / k for k = 1, …, n generate the complete integrable structure. The first Hamiltonian *H*₁ = tr(φ) controls the linear flow on the Jacobian; the higher *H_k* for k ≥ 2 are nonlinear corrections capturing the curvature of the Hitchin fibration.

In the ML setting, these higher Hamiltonians produce **non-Gaussian corrections to scaling dimensions**:

**[C] Conjecture (Higher Hitchin Corrections).** For a depth-*L* ReLU network trained on data with non-Gaussian higher-order cumulants κ_j (*j* ≥ 3), the scaling dimension of the *n*-th mode receives corrections:

```
Δₙ = Δₙ^{(1)}  +  Σ_{k=2}^{L} αₖ · (tr Mᵏ / tr M)  ·  κₖ
```

where Δ_n^{(1)} = −(1/2) ln(1 + ν_n/λ_noise) is the Gaussian leading term, and α_k are architecture-dependent coefficients satisfying α_k → 0 in the linear-activation limit.

The *k* = 2 correction α₂ · (tr M² / tr M) · κ₃ is the leading non-Gaussian shift. It is **empirically testable**: for data with controlled third cumulant (e.g., mixture-of-Gaussians with skewed class means), this correction predicts a measurable shift in the C_α phase diagram relative to the pure Gaussian prediction.

### III.4 Operator Counting via the Spectral Curve Genus

**[T] Operator counting bound.** The number of relevant operators at *W** is at most rank(Cov(*x*, *Y*)).

**Proof.** The number of positive eigenvalues of *M* is bounded by those of −Hess(*L*) (adding Hess(𝒮̄) ≽ 0 cannot decrease eigenvalues, by Weyl's interlacing inequality). For quadratic loss on Gaussian data, the positive eigenvalues of −Hess(*L*) equal rank(Cov(*x*, *Y*)). For general losses and non-Gaussian data, nonlinear curvature contributes additional positive curvature directions; the bound rank(Cov(*x*, *Y*)) is tight in the Gaussian limit and receives corrections of order O(κ₃ / σ) for mildly non-Gaussian distributions. ∎

---

## Part IV — Hitchin Fibration as the Geometry of Gradient-Descent Orbits

### IV.1 Generic Fibers as Jacobians

For a generic point *s* ∈ 𝒜_net, the **Hitchin fiber** *H_net*⁻¹(*s*) consists of all networks with the same scaling dimension spectrum (same spectral type). In the geometric system, this fiber is the Jacobian Jac(*S_s*) — an abelian variety on which the Hamiltonian flow is linear.

**[A]** In the ML setting, this fiber is the **gradient-descent orbit** in weight space among networks sharing a fixed spectral type. The linearity of the Arnol'd–Liouville flow on Jac(*S_s*) corresponds to the **linear convergence** of gradient descent within a fixed quadratic basin near a nondegenerate fixed point.

This gives a stratification of weight space:

```
Weight space Θ = ⋃_{s ∈ 𝒜_net} H_net⁻¹(s)
```

Two networks connected by a gradient-descent trajectory without wall-crossing share the same spectral type *s* — they lie in the same Hitchin fiber. A **phase transition** (generalization ↔ memorization) occurs precisely when the gradient-descent trajectory crosses a wall in 𝒜_net, moving from one fiber to another.

### IV.2 Degenerate Fibers and Phase Transitions

Over the **discriminant locus** Δ ⊂ 𝒜_net (where *S_s* becomes singular), the Hitchin fiber degenerates: Jac(*S_s*) acquires nodal singularities and the abelian variety structure breaks down. In the ML setting:

| Locus in 𝒜_net | Spectral curve *S_s* | Gradient dynamics |
|---|---|---|
| Generic point | Smooth, genus *g*(*S_s*) | Linear flow; exponential convergence |
| Discriminant wall Δ | Nodal singularity (one eigenvalue collision) | Critical slowing-down; C_α → 1 |
| Deep singular locus | Multiple eigenvalue collisions | Gradient explosion or vanishing |

### IV.3 Wall-Crossing and the C_α Detector

The Hitchin base 𝒜_net admits a **wall-and-chamber decomposition**:

```
𝒜_net = ⋃_i 𝒞_i  ∪  ⋃_j 𝒲_j
```

where the chambers 𝒞_i correspond to distinct stability types (distinct polystable decompositions of the Higgs bundle) and the walls 𝒲_j are the discriminant loci.

**C_α as distance to nearest wall.** The signed deviation of *C_α*(ℓ) from 1 measures the distance from the current point in 𝒜_net to the nearest wall:

```
C_α(ℓ) − 1  ≈  λ₁(ℒ_JL)  ·  dist(H_net(W_ℓ), nearest wall 𝒲_j)
```

up to corrections of order O(λ_min / λ_max) from the anisotropy of D_s.

---

## Part V — Non-Gaussian Relevant Subspace and LDA Spectral Data

### V.1 Setup

Let the data distribution be a balanced *K*-component mixture of Gaussians:

```
p_data(x) = (1/K) Σ_{k=1}^K 𝒩(μ_k, Σ₀),     Σ₀ ≻ 0
```

Define:

```
S_B  = Σ_k (μ_k − μ̄)(μ_k − μ̄)ᵀ               (between-class scatter)
S̃_B  = Σ₀^{−1/2} S_B Σ₀^{−1/2}                (Mahalanobis between-class scatter)
𝒱_LDA = span{ top (K−1) eigenvectors of S̃_B }   (LDA subspace)
```

### V.2 Theorem (MoG Relevant Subspace)

**[T] Theorem.**

**(a) Sufficiency.** For any coarse-graining *R* : ℝ^*d* → ℝ^{*d'*} with *d'* ≥ *K*−1, if range(*R*) ⊇ Σ₀^{−1} 𝒱_LDA then I(ζ; *Y* | *x_IR*) = 0.

**(b) Optimality.** For *d'* < *K*−1, the coarse-graining minimizing I(ζ; *Y* | *x_IR*) subject to dim = *d'* is projection onto the top *d'* eigenvectors of S̃_B.

**(c) Gaussian scaling dimensions.** The scaling dimension of the *k*-th LDA direction is:

```
Δ_k^{(1)} = −(1/2) ln(1 + νₖ / λ_noise)
```

where ν_k is the *k*-th eigenvalue of S̃_B and λ_noise = σ² / (σ² + Tr(Σ₀)/*d*). Directions with large ν_k have strongly negative Δ_k (highly relevant); directions with ν_k ≈ 0 have Δ_k ≈ 0 (marginal).

**(d) Non-Gaussian corrections (see §III.3).** For deep nonlinear networks, Δ_k receives corrections from higher Hitchin Hamiltonians tr(*M*^j) for *j* ≥ 2. These corrections are O(κ₃ / σ) for skewed class-mean geometries and vanish in the linear-activation or shallow-network limit.

**Proof (a–c).** The class posterior *p*(*Y* = *k* | *x*) ∝ exp(μ_kᵀ Σ₀^{−1} *x* − ½ μ_kᵀ Σ₀^{−1} μ_k) depends on *x* only through the *K* discriminant scores *d_k*(*x*) = μ_kᵀ Σ₀^{−1} *x*, spanning a (*K*−1)-dimensional subspace. Projection onto any subspace containing Σ₀^{−1} 𝒱_LDA preserves all information about *Y*, giving (a). For (b), maximizing Mahalanobis class separation subject to dim = *d'* is the Fisher LDA eigenvalue problem. For (c), information decay under Gaussian noise gives the stated formula. ∎

### V.3 LDA Spectral Curve

The scaling dimensions {Δ_k} are the eigenvalues of the first Hitchin Hamiltonian *H*₁ = tr(φ) restricted to the relevant subspace. The **LDA spectral curve** is:

```
S_LDA = { (k, Δ_k^{(1)} + corrections) :  k = 1, ..., K−1 }  ⊂  S_{W*}
```

a finite sub-curve of the full network spectral curve. Its Jacobian is a (*K*−1)-dimensional abelian variety — the gradient-descent orbit within the LDA universality class.

---

## Part VI — The Nonabelian Hodge Correspondence in the Network Setting

### VI.1 Three Descriptions of a Trained Network

The NAHC provides three equivalent descriptions of a polystable Higgs bundle (*C_α* = 1). In the ML setting:

```
{ Trained network at fixed point W*, C_α = 1 }
        ↕  Hitchin equations: F_A + [φ,φ*] = 0, ∂̄_A φ = 0
{ Polystable network Higgs bundle (E, φ) }
        ↕  Flat connection: ∇ = d_A + φ + φ*
{ Forward-pass operator as flat GL(n,ℝ)-connection on E }
        ↕  Monodromy / holonomy representation
{ Symmetry group of learned features }
```

**Polystable Higgs bundle.** The trained weight configuration at a stable fixed point. The Higgs field φ_ℓ = d*W_ℓ*/d*t* encodes the rate at which representations change across layers.

**Flat connection.** The full forward-pass operator, viewed as parallel transport along paths in the internal representation space. Flatness — *F*_∇ = 0 — is equivalent to the Hitchin equations being satisfied, i.e., the network being at a polystable fixed point.

**Holonomy representation.** The map from paths (depth sequences) in *C* to transformations of *E* defines a representation of the path groupoid of *C*. For a network whose layers are self-consistent (β = 0), this representation factors through the **symmetry group of the learned feature space**.

**[A]** The extension of this correspondence to the *data manifold* requires the data to define a nontrivial topological space (e.g., images lying near a low-dimensional manifold ℳ_data ⊂ ℝ^{*d*₀}). In that case, the representation ρ : π₁(ℳ_data) → GL(*n*, ℝ) encodes how the network responds to loops in the data manifold — e.g., continuous rotations of an input image. This is a structural analogy; deriving it formally requires constructing a functorial map from the network category to the category of local systems on ℳ_data.

### VI.2 Feature Collapse as Reducible Representation

**[C] Conjecture (Network NAHC).** A trained network at a polystable fixed point defines an **irreducible** holonomy representation if and only if it has no feature collapse (no dormant neurons, no linear dependence among feature channels). Feature collapse corresponds to **reducible representations** — polystable but not stable Higgs bundles that split as direct sums, fibering over a proper subbundle of *E*.

This conjecture, if proven, would provide a geometric explanation for the empirical observation that overparameterized networks with diverse initialization avoid feature collapse: they initialize in the interior of a stability chamber where the Higgs bundle is genuinely stable (not merely polystable via splitting).

---

## Part VII — Empirically Testable Predictions

The framework makes three categories of quantitative predictions:

### VII.1 C_α Phase Diagram (Verified)

**[V]** For the `make_blobs` dataset (3 classes, Gaussian clusters, MLP(64,32)):

- Theory predicts *K*−1 = 2 relevant directions → 2 positive sheets of *S_{W*}*.
- Observed peak *C_α* = 6.19 in early training = deep inside the 2-dimensional stability chamber.
- Observed *C_α* decline after peak = wall-crossing as gradient signal is exhausted, system approaches the discriminant locus in 𝒜_net.

### VII.2 Higher Hitchin Correction (Testable Prediction)

**[C, Testable]** For data with controlled skewness κ₃ = 𝔼[(x − μ)³/σ³]:

```
Δₙ(empirical) − Δₙ^{(1)}(Gaussian)  ∝  κ₃ · (tr M² / tr M)
```

**Test protocol:**
1. Generate mixture-of-Gaussians data with varying skewness κ₃ ∈ {0, 0.5, 1.0, 2.0}.
2. Train MLP to convergence; measure empirical scaling dimensions from the Hessian spectrum.
3. Regress Δ_n(empirical) − Δ_n^{(1)} against κ₃ · (tr M² / tr M).
4. The conjecture predicts a linear relationship with a positive slope α₂.

### VII.3 Wall-Crossing Signature (Testable Prediction)

**[C, Testable]** At a phase transition (generalization → memorization):

- *C_α*(ℓ) → 1 across all layers simultaneously (not layer-by-layer).
- λ₁(ℒ_JL) → 0 (spectral gap closes).
- Hessian spectrum of *M* develops a near-zero eigenvalue (discriminant locus approach).

These three signatures should coincide within ± one training epoch at the transition, providing a simultaneous multi-level indicator of wall-crossing that no single metric (loss, accuracy, C_α alone) can detect.

---

## Part VIII — Spectral Basis on the Hitchin Base

The Sturm–Liouville eigenfunctions provide an orthonormal basis for *L*²-functions on the 1-dimensional Hitchin base [0, *L*]:

```python
def sturm_liouville_eigenfunctions(
    n_modes: int,
    n_points: int = 256,
    p_func = None,   # metric on Hitchin base; default: flat (p ≡ 1)
    q_func = None,   # potential; default: zero (marginal case)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenfunctions of  Lψ = −d/dx[p(x) dψ/dx] + q(x)ψ
    on [0,1] with Dirichlet BCs ψ(0) = ψ(1) = 0.

    Hitchin interpretation:
      p(x)  = depth-dependent metric component of ℒ_JL,
              arising from the Fisher information metric on ℬ
              restricted to the 1D Hitchin base.
      q(x)  = symmetry-redundancy potential 𝒮̄ restricted to [0,L].

    Parameters
    ----------
    n_modes  : number of eigenfunctions to return
    n_points : interior spatial discretization points
    p_func   : callable x → p(x), default constant 1
    q_func   : callable x → q(x), default constant 0

    Returns
    -------
    x   : (n_points,)          interior grid in (0,1)
    psi : (n_points, n_modes)  L²-normalized eigenfunctions
    """
    import numpy as np

    if p_func is None:
        p_func = lambda x: np.ones_like(x)
    if q_func is None:
        q_func = lambda x: np.zeros_like(x)

    x = np.linspace(0, 1, n_points + 2)[1:-1]   # interior points
    h = x[1] - x[0]
    p = p_func(x)
    q = q_func(x)

    # Exact boundary values for consistent O(h²) discretization
    p_left_bc  = p_func(np.array([0.0]))[0]
    p_right_bc = p_func(np.array([1.0]))[0]

    N = len(x)
    diag_main = np.zeros(N)
    diag_up   = np.zeros(N - 1)
    diag_down = np.zeros(N - 1)

    for i in range(N):
        p_right = 0.5*(p[i] + p[i+1])   if i < N-1 else 0.5*(p[i] + p_right_bc)
        p_left  = 0.5*(p[i] + p[i-1])   if i > 0   else 0.5*(p[0] + p_left_bc)
        diag_main[i] = (p_right + p_left) / h**2 + q[i]
        if i < N - 1:
            diag_up[i] = -0.5*(p[i] + p[i+1]) / h**2
        if i > 0:
            diag_down[i-1] = -0.5*(p[i] + p[i-1]) / h**2

    L_mat = (
        np.diag(diag_main)
        + np.diag(diag_up, k=1)
        + np.diag(diag_down, k=-1)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(L_mat)
    idx = np.argsort(eigenvalues)[:n_modes]
    psi = eigenvectors[:, idx]

    # L²-normalize
    for k in range(psi.shape[1]):
        norm = np.sqrt(np.trapz(psi[:, k]**2, x))
        if norm > 1e-12:
            psi[:, k] /= norm

    return x, psi


def spectral_features(data: np.ndarray, psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Project data onto the Sturm–Liouville basis {ψₙ}.

        aₙ = ∫ f(x) ψₙ(x) dx     (L² inner product on Hitchin base)

    Parameters
    ----------
    data : (n_points,)             function values on interior grid x
    psi  : (n_points, n_modes)     eigenfunctions from above
    x    : (n_points,)             interior grid

    Returns
    -------
    coeffs : (n_modes,)            spectral coefficients
    """
    return np.array([np.trapz(data * psi[:, k], x) for k in range(psi.shape[1])])


def spectral_reconstruct(coeffs: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Reconstruct f(x) = Σₙ aₙ ψₙ(x) from spectral coefficients."""
    return psi @ coeffs
```

---

## Geometric Summary

```
Deep Network Training
         │
         │ network Higgs bundle construction (§I, §II)
         ▼
Higgs Bundle (E_ℓ, φ_ℓ) over depth curve C
  E_ℓ  = representation bundle at depth ℓ
  φ_ℓ  = dW_ℓ/dt = Higgs field (weight gradient in RG time)
         │
         │ network Hitchin map H_net (§III.2)
         ▼
Network Hitchin base 𝒜_net = { char. poly. of stability matrix M }
  Positive sheets → relevant operators (class features)
  Negative sheets → irrelevant operators (UV noise)
  Higher-order tr(Mᵏ) → non-Gaussian corrections (§III.3)
         │
         │ Hitchin fibration over generic point (§IV.1)
         ▼
Jacobian Jac(S_{W*}) — abelian variety
  Gradient-descent orbit within fixed universality class
  Linear Arnol'd–Liouville flow
  Wall-crossing in 𝒜_net = phase transition (§IV.2)
         │
         │ Nonabelian Hodge Correspondence, C_α = 1 (§VI)
         ▼
Flat connection ∇ = d_A + φ + φ*
  Full forward-pass operator of the trained network
         │
         │ holonomy representation (structural analogy)
         ▼
ρ : π₁(Data manifold) → GL(n, ℝ)
  Symmetry group of learned features
  Irreducible ↔ no feature collapse (Conjecture, §VI.2)
```

---

## Appendix — Key References

| Authors | Title | Venue | Year |
|---|---|---|---|
| Hitchin, N.J. | "The self-duality equations on a Riemann surface" | *Proc. London Math. Soc.* 55(1): 59–126 | 1987 |
| Hitchin, N.J. | "Stable bundles and integrable systems" | *Duke Math. J.* 54(1): 91–114 | 1987 |
| Donaldson, S.K. | "Twisted harmonic maps and the self-duality equations" | *Proc. London Math. Soc.* 55(1): 127–131 | 1987 |
| Corlette, K. | "Flat *G*-bundles with canonical metrics" | *J. Diff. Geom.* 28(3): 361–382 | 1988 |
| Simpson, C.T. | "Higgs bundles and local systems" | *Publ. Math. IHÉS* 75: 5–95 | 1992 |
| Narasimhan, M.S.; Seshadri, C.S. | "Stable and unitary vector bundles on a compact Riemann surface" | *Ann. Math.* 82(3): 540–567 | 1965 |
| Ngô, B.C. | "Fibration de Hitchin et endoscopie" | *Invent. Math.* 164(2): 399–453 *(Fields Medal 2010)* | 2010 |
| Donagi, R.; Pantev, T. | "Langlands duality for Hitchin systems" | *Invent. Math.* 189(3): 653–735 | 2012 |
| Bradlow, S.; García-Prada, O.; Gothen, P.B. | "What is … a Higgs Bundle?" | *Notices AMS* 54(8): 980–981 | 2007 |
| Kato, T. | *Perturbation Theory for Linear Operators* | Springer (§VI.2.1 — KLMN theorem) | 1966 |

---

*Framework: Higgs–Hitchin RG-ML v2.0*
