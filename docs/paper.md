# Certified stochastic stability at scale: factorised Lyapunov verification for high-dimensional physical systems

## Abstract

Deploying learned controllers and monitors on physical infrastructure requires stability certificates, yet classical Lyapunov certification scales exponentially with system dimension — the curse of dimensionality that keeps rigorous guarantees out of the very systems that need them most. Here we introduce a factorised certification scheme that separates a high-dimensional physical state into a low-dimensional certified subspace of dimension $d$ and an orthogonal residual, and verifies the stochastic Lyapunov inequality over the *entire* region by interval branch-and-bound in the factorised coordinates. The scheme is sound by construction (Theorem 1), complete under refinement (Theorem 2), and robust to identification error in the plant model (Theorem 3). An invertible neural transport learns coordinates in which the certified subspace captures the dominant dynamics; on a nonlinear 8-dimensional plant where principal-component and random projections certify 2.0% and 0.0% of the operating region, the learned transport certifies 64.9% with certified bounds two orders of magnitude tighter. Scaling ladders from $D=8$ to $D=200$ hold certification cost fixed in $d$ while the full-dimensional verifier certifies nothing at any budget. We validate the pipeline on three real datasets (turbofan degradation, electricity-transformer telemetry, a 1404-channel chemical-reactor sensor array) and on a fluid-mechanics plant — the Kármán vortex street evolved by Chorin's random vortex method — and demonstrate that a shadow-network gate built on the certificate rejects every unsafe controller swap on a real regime change while remaining sound. The result is a complete, reproducible pipeline — identification, transport learning, certificate training, sound verification, and gated deployment — that converts certification from an exponentially hard analytic task into a learned-then-verified one.

## Introduction

Every autonomous system that touches the physical world — a turbine controller, a grid stabiliser, a fluid monitor — carries an implicit promise: that its behaviour near the operating point is stable under noise. For linear systems the promise is cheap (spectral radii); for the nonlinear, stochastically forced, partially observed systems that actually get deployed, it is a Lyapunov certificate, and nobody can compute one at scale. Sum-of-squares programming scales poorly past a dozen dimensions; SMT-based neural verifiers inherit the same exponential blow-up in the state dimension $D$. The consequence is structural: certified deployment stops at toy scale precisely where the engineering stakes get real.

The contribution of this work is a change of coordinates. We do not certify the physical state; we certify a *learned, invertible transport* $T$ of it. The transport splits the state into a $d$-dimensional certified block $\eta$ (carrying the Lyapunov certificate $V(\eta)$) and a residual block $\rho$ (carrying a fixed quadratic form). Because the Lyapunov function and its generator are verified in the factorised coordinates, the branch-and-bound search that proves the certificate over a continuous region costs $\mathrm{poly}(d)$ per node rather than $\mathrm{poly}(D)$ — and the soundness of the split is a theorem, not a heuristic (Theorem 1: the factorised bound is a valid upper bound of the true generator; Theorem 2: the search converges to the true supremum under box refinement; Theorem 3: the certificate tolerates plant model error up to an explicit margin).

This bridges two communities that rarely meet. The learning community builds expressive dynamics models but validates them by sampling; the verification community produces sound certificates but at dimensions far below real plants. The pipeline here is deliberately both: every neural component is trained against the exact stochastic generator (where an exact one exists) or against identified plants with bootstrap-quantified identification error (where it does not), and every deployment claim is certified by interval arithmetic over the *whole* region — never by sampled validation alone.

## Results

### A sound factorised certificate

The certificate is $W(\eta,\rho) = V(\eta) + \kappa\,\rho^\top Q \rho$ on the transported coordinates, and the claim verified over the region is the generator inequality $\sup ( \mathcal{L}W + \alpha W ) \le \beta + \mathrm{tol}$ with $\beta = \mathcal{L}W(0)$ the measured noise floor. The factorised verifier bounds the transport, its interval Jacobian, and the latent drift on interval hulls — sound by outward-rounded interval arithmetic, and tight enough to certify (full derivations and proofs: [the formal supplement](math.html), Theorems 1–4).

### The learned transport is what makes nonlinear certification possible

On the 4-link arm ($D=8$), where the coordinate map must undo the dynamics' own trigonometric curvature, the three coordinate systems separate decisively — this is the experiment that turns the transport from an implementation detail into the enabling mechanism:

| coordinate map | region certified | worst certified bound |
|---|---|---|
| **learned invertible transport** | **64.9%** | 0.20 |
| PCA | 2.0% | 11.4 |
| random orthogonal | 0.0% | 49.3 |

The complement matters as much as the headline: on linear-Gaussian identified plants (exp7) *any* orthogonal map certifies 100% — the machinery alone suffices there — so the transport's contribution is exactly the nonlinear regime, where it is decisive.

### Certification cost is flat in $D$ when it is set by $d$

The scaling ladder (exp1) saturates at $D=8$ and the supremacy ladder (exp10) extends it to $D=200$: the factorised verifier's cost per node and its node budgets are set by the latent dimension $d=2$, while the full-dimensional verifier — the honest baseline — certifies 0% at every rung regardless of budget. Scaling-law data and per-arm wall-clock numbers: [results, section 10](results.html#supremacy).

### Real domains, including a negative result

The pipeline runs end-to-end on real telemetry: turbofan degradation (C-MAPSS), air quality, and an electricity-transformer grid (ETTm2), plus a 1404-channel chemical-reactor sensor array at $D=176$ (exp11) — the first real domain above $D=100$. We report a genuine negative result: on the noise-dominated ETTm2 plant the pointwise certificate closes at 0% certified volume, with the mechanism identified (per-cycle noise relative to drift exceeds the certificate's separation margin), while the shadow gate on the same plant remains perfectly sound — zero unsafe swaps against a naive gate's eight. Three domains of heterogeneous difficulty are the honest calibration of what the method certifies and what it cannot; the reactor-scale era probes quantify exactly what a real regime change does to a deployed certificate ([results, section 11](results.html#reactor)).

### A fluid-mechanics grand-challenge plant

The fluid domain (exp12) is the Kármán vortex street evolved by Chorin's random vortex method — the classical particle discretisation of 2D Navier–Stokes: $n=50$ vortices ($D=100$) advected by the exact Biot–Savart kernel with Lamb–Oseen cores, closed by window flushing, forced by the Brownian core walk $\sqrt{2\nu}\,dW$ that models viscous diffusion. The drift plant moves the street off its stability optimum ($b/h$: 0.281 → 0.45) and into an $8\times$ more diffusive regime. This is the first plant in the suite whose drift is a genuine nonlinear fluid-mechanics operator rather than a mechanical or identified-linear model ([results, section 12](results.html#fluid)).

### Statistical rigor as a first-class output

Multi-seed stability of the headline separation, bootstrap confidence intervals over region probes (2000 resamples), identification floors on held-out data halves, and ablations that confirm $\beta$ tracks the diffusion scaling exactly as the theory predicts: [results, section 13](results.html#rigor).

## Discussion

The factorisation is a trade, stated plainly: certification is exact only in the $d$-dimensional certified block, and the residual block is covered by a fixed quadratic form whose conservatism grows with residual energy. The d-not-D phenomenon — full-dim certifies nothing at any budget while factorised mode certifies real volume — is both the method's power and its honest boundary: the certificate is as good as the transport's ability to concentrate dynamics into $d$ coordinates. Where noise dominates drift (ETTm2), no coordinate change rescues the margin, and the pipeline says so rather than certifying an illusion. Theorem 3's robustness margin converts that diagnosis into a deployment rule: the certificate remains valid for plants within $\varepsilon$ of the identified one, and the identification floor measures $\varepsilon$ per era.

## Methods overview

Identify (era-wise linear-Gaussian fits with held-out floors) → learn the transport (invertible affine-coupling blocks with $T(x^\star)=0$) → fit the latent world model with the two-stage push-forward refit (frozen-transport second stage; collusion-free) → train the quadratic-led Lyapunov net against exact-generator violations → measure the noise floor $\beta$ → certify by interval branch-and-bound at fixed node budgets → gate deployment with the dual-buffer shadow network. Full mathematics, theorems and proofs: [the supplement](math.html). Every experiment reproduces from committed JSONs and pinned seeds: [reproduce list](results.html#reproduce).

## Data availability

C-MAPSS turbofan, air-quality, and ETTm2 transformer telemetry: public Kaggle mirrors, loaders in `sbounds/realsys.py`. Chemical-reactor domain-adaptation array: Kaggle `eddardd/continuous-stirred-tank-reactor-domain-adaptation`. All generated plant data (arms, vortex street) is produced by pinned seeds in the repository.

## Code availability

The full pipeline, interval kernel, verifier, and all experiment drivers are in the repository; each results section states its driver and budget.
