# UDiff Methodology

> Mathematical foundations and algorithmic details of the Deep Unfolded Diffusion Recovery framework.

---

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Diffusion Model Background](#diffusion-model-background)
3. [Deep Unfolding Framework](#deep-unfolding-framework)
4. [Sherman–Morrison–Woodbury Derivation](#shermanmorrisonwoodbury-derivation)
5. [End-to-End Training with Deep Supervision](#end-to-end-training-with-deep-supervision)

---

## Problem Formulation

### Compressed Sensing Setup

We consider the standard compressed sensing problem of recovering a sparse signal **x** ∈ ℝⁿ from a set of linear measurements:

```
y = Φx + n
```

where:
- **y** ∈ ℝᵐ is the measurement vector (m ≪ n)
- **Φ** ∈ ℝᵐˣⁿ is the sensing (measurement) matrix
- **x** ∈ ℝⁿ is the unknown sparse signal to recover
- **n** ∈ ℝᵐ is additive measurement noise, typically n ~ 𝒩(0, σ²I)

### Sparsity Model

The signal **x** follows a Bernoulli–Gaussian (BG) model:

```
x_i = b_i · g_i,     i = 1, ..., n
```

where:
- b_i ~ Bernoulli(ρ) with sparsity rate ρ (e.g., ρ = 0.1)
- g_i ~ 𝒩(0, σ_x²) are independent Gaussian amplitudes

The compression ratio is defined as δ = m/n, where m is the number of measurements. The goal is to recover **x** from **y** when δ < 1.

### Classical Optimization Perspective

Sparse recovery can be formulated as:

```
minimize  (1/2) ||y - Φx||²₂ + τ · R(x)
```

where R(x) is a sparsity-promoting regularizer (e.g., ℓ₁ norm for LASSO, or a learned prior). Traditional algorithms (ISTA, ADMM) solve this iteratively, with each iteration performing:

1. **Gradient/proximal step**: Move toward data fidelity
2. **Denoising/thresholding step**: Enforce the prior/regularizer

UDiff replaces the hand-crafted denoiser with a learned diffusion-based denoiser, embedded within a principled deep unfolding framework.

---

## Diffusion Model Background

### Forward Diffusion Process

A diffusion probabilistic model defines a forward Markov chain that gradually adds Gaussian noise to data over T timesteps:

```
q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1 - βₜ) · xₜ₋₁, βₜ · I)
```

where {βₜ}ₜ₌₁ᵀ is a variance schedule with 0 < βₜ < 1. Using the notation αₜ = 1 - βₜ and ᾱₜ = ∏ₛ₌₁ᵗ αₛ, we can express any noisy version directly:

```
q(xₜ | x₀) = 𝒩(xₜ; √ᾱₜ · x₀, (1 - ᾱₜ) · I)
```

This means: `xₜ = √ᾱₜ · x₀ + √(1 - ᾱₜ) · ε`, where ε ~ 𝒩(0, I).

### Reverse Diffusion (Denoising)

The reverse process learns to denoise:

```
p_θ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μ_θ(xₜ, t), σₜ² · I)
```

The model can equivalently be trained to predict:
- The clean signal x₀ (x₀-prediction)
- The noise ε (ε-prediction)
- The score ∇ₓ log p(xₜ) (score prediction)

For UDiff, we use **x₀-prediction**: the denoiser D_θ(xₜ, t) directly estimates x₀ from xₜ.

### Training Objective

The standard diffusion training loss is:

```
L_diffusion = 𝔼_{x₀, t, ε} [ ||D_θ(√ᾱₜ · x₀ + √(1-ᾱₜ) · ε, t) - x₀||²₂ ]
```

### Connection to Denoising

At noise level β, the optimal denoiser in the MMSE sense is the posterior mean:

```
D*(z, β) = 𝔼[x | z] = 𝔼[x | x + √β · ε]
```

The diffusion denoiser D_θ approximates this posterior mean, providing a powerful learned prior for signal recovery.

---

## Deep Unfolding Framework

### Motivation

Classical iterative algorithms (ISTA, ADMM) alternate between:
1. A **data fidelity** step (measurement consistency)
2. A **regularization** step (denoising/thresholding)

Deep unfolding "unrolls" K iterations of such an algorithm into a K-layer neural network, replacing hand-crafted components with learned modules and making all hyperparameters learnable.

### UDiff Algorithm

UDiff unrolls K stages. At each stage k = 1, ..., K:

**Step 1: Measurement Consistency**

```
z_k = x̂_{k-1} + λ_k · Φᵀ(ΦΦᵀ + μI)⁻¹(y - Φx̂_{k-1})
```

This is a closed-form proximal update that projects the current estimate toward measurement consistency. The term `(y - Φx̂_{k-1})` is the measurement residual, and the operator `Φᵀ(ΦΦᵀ + μI)⁻¹` maps it back to signal space with Tikhonov regularization.

**Step 2: Diffusion Denoising**

```
x̂_k = η_k · D_θ(z_k, β_k) + (1 - η_k) · z_k
```

The denoiser D_θ processes the measurement-consistent estimate z_k at noise level β_k. The relaxation factor η_k interpolates between the denoiser output and the input.

### Initialization

```
x̂₀ = Φᵀy    (matched filter / backprojection)
```

### Complete Algorithm

```
Algorithm: UDiff Forward Pass
─────────────────────────────
Input:  y (measurements), Φ (sensing matrix)
Params: θ (denoiser weights), {β_k, λ_k, η_k}_{k=1}^K

1. Precompute: G = (ΦΦᵀ + μI)⁻¹
2. Initialize: x̂₀ = Φᵀy

3. For k = 1, ..., K:
   3a. r_k = y - Φ · x̂_{k-1}              # Measurement residual
   3b. z_k = x̂_{k-1} + λ_k · ΦᵀG · r_k   # Consistency step
   3c. d_k = D_θ(z_k, β_k)                 # Diffusion denoiser
   3d. x̂_k = η_k · d_k + (1 - η_k) · z_k  # Relaxation

4. Return x̂_K
```

### Key Design Choices

1. **Weight sharing**: The denoiser D_θ shares weights across all K stages. Only the noise level β_k varies, meaning the same network denoises at progressively lower noise levels.

2. **Decreasing noise schedule**: The noise levels {β_k} are initialized as a decreasing sequence (coarse-to-fine recovery). After end-to-end training, they typically remain monotonically decreasing.

3. **Closed-form consistency**: Unlike methods that require iterative inner loops for the consistency step, UDiff uses a closed-form expression enabled by the SMW identity.

---

## Sherman–Morrison–Woodbury Derivation

### The Matrix Inversion Challenge

The measurement consistency step requires computing:

```
Φᵀ(ΦΦᵀ + μI_m)⁻¹
```

Naively, one might consider working in the signal domain:

```
(ΦᵀΦ + μI_n)⁻¹Φᵀ
```

which requires inverting an n × n matrix — expensive when n is large.

### Woodbury Identity

The Woodbury matrix identity states:

```
(A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
```

### Application to Our Setting

We want to compute `(ΦᵀΦ + μI_n)⁻¹Φᵀ`. Setting A = μI_n, U = Φᵀ, C = I_m, V = Φ:

```
(μI_n + ΦᵀΦ)⁻¹ = (1/μ)I_n - (1/μ)Φᵀ(I_m + (1/μ)ΦΦᵀ)⁻¹(1/μ)Φ
                 = (1/μ)I_n - (1/μ²)Φᵀ(I_m + (1/μ)ΦΦᵀ)⁻¹Φ
```

Multiplying by Φᵀ on the right side for our consistency step:

```
(μI_n + ΦᵀΦ)⁻¹Φᵀ = (1/μ)Φᵀ - (1/μ²)Φᵀ(I_m + (1/μ)ΦΦᵀ)⁻¹ΦΦᵀ
```

Simplifying with the push-through identity:

```
(μI_n + ΦᵀΦ)⁻¹Φᵀ = Φᵀ(ΦΦᵀ + μI_m)⁻¹
```

### Computational Benefit

| Approach | Matrix to invert | Size | Cost |
|----------|-----------------|------|------|
| Signal domain | ΦᵀΦ + μI_n | n × n | O(n³) |
| **Measurement domain (SMW)** | **ΦΦᵀ + μI_m** | **m × m** | **O(m³)** |

Since m ≪ n in compressed sensing, the SMW approach provides a significant computational advantage. For typical settings (n = 256, m = 64), this yields a **64× reduction** in inversion cost.

### Precomputation Strategy

The matrix `G = (ΦΦᵀ + μI_m)⁻¹` depends only on Φ and μ, both of which are fixed during inference. Therefore:

1. Compute `G = (ΦΦᵀ + μI_m)⁻¹` once at initialization
2. Precompute `ΦᵀG` (an n × m matrix)
3. At each stage, the consistency update reduces to a matrix-vector product: `ΦᵀG · r_k`

This makes each stage's consistency step O(nm) — a simple matrix-vector multiply.

---

## End-to-End Training with Deep Supervision

### Training Objective

UDiff is trained end-to-end by minimizing a weighted sum of reconstruction losses across all K stages:

```
L_total = Σ_{k=1}^{K} w_k · L_k(x̂_k, x)
```

where:
- `x̂_k` is the estimate at stage k
- `x` is the ground-truth sparse signal
- `L_k` is the per-stage loss
- `w_k` is the stage weight

### Per-Stage Loss

The per-stage loss combines MSE and a measurement consistency term:

```
L_k = ||x̂_k - x||²₂ + α · ||y - Φx̂_k||²₂
```

where α ≥ 0 balances reconstruction accuracy with measurement fidelity.

### Deep Supervision Weights

Stage weights increase linearly to emphasize later (more refined) stages:

```
w_k = k / Σ_{j=1}^{K} j = 2k / (K(K+1))
```

For K = 8: w₁ = 1/36, w₂ = 2/36, ..., w₈ = 8/36.

### Learnable Parameters

The full set of learnable parameters is:

```
Θ = {θ, {β̃_k, λ̃_k, η̃_k}_{k=1}^{K}}
```

where:
- θ: Denoiser network weights (shared across stages)
- β̃_k: Unconstrained noise level parameters → β_k = sigmoid(β̃_k)
- λ̃_k: Unconstrained consistency weights → λ_k = softplus(λ̃_k)
- η̃_k: Unconstrained relaxation factors → η_k = sigmoid(η̃_k)

### Gradient Flow

End-to-end training requires gradients to flow through:
1. The denoiser D_θ (standard backpropagation)
2. The consistency step (fully differentiable linear operations)
3. The parameter transformations (sigmoid, softplus — both differentiable)

The deep supervision loss provides gradient signal at every stage, mitigating vanishing gradients in deep unrolled architectures.

### Training Procedure

```
Algorithm: UDiff Training
──────────────────────────
1. Initialize denoiser θ (random or pretrained)
2. Initialize {β̃_k} as linearly decreasing logits
3. Initialize {λ̃_k} as softplus⁻¹(1.0) for all k
4. Initialize {η̃_k} as sigmoid⁻¹(0.5) for all k

5. For each epoch:
   5a. Sample batch of sparse signals x ~ BG(ρ, σ_x²)
   5b. Generate measurements y = Φx + n
   5c. Forward pass through K stages → {x̂₁, ..., x̂_K}
   5d. Compute deep supervision loss L_total
   5e. Backpropagate and update Θ via Adam optimizer

6. Learning rate schedule: cosine annealing with warm restarts
7. Optional: gradient clipping with max norm 1.0
```

### Convergence Properties

- The measurement consistency step is a contraction mapping (for appropriate λ_k), ensuring stability.
- Deep supervision prevents gradient vanishing across stages.
- The decreasing noise schedule provides a natural coarse-to-fine curriculum.
- Weight sharing reduces parameter count and improves generalization.
