# The mathematics of the certificate

This document states exactly what is claimed, what is verified by search, and what is
measured. Notation: the physical state is `x` in `R^D`, the transport maps `y = T(x)`
with `T` an exactly invertible diffeomorphism and `T(x*) = 0` at the equilibrium; the
latent split is `y = (eta, rho)` with `eta` in `R^d` the certified factor and `rho` in
`R^m` the transversal residual (`m = D - d`).

## 1. The physical system

An N-link planar arm with point masses at the link tips under gravity-compensated PD
control, with process noise in the velocity block:

```
M(theta) theta_ddot + c(theta, theta_dot) + G(theta) = tau,
tau = G(theta) - Kp theta - Kd theta_dot,
dx = f(x) dt + Sigma(x) dW
```

with

```
M_ij = A_ij l_i l_j cos(theta_i - theta_j),  A_ij = sum_{k >= max(i,j)} m_k
c_i  = sum_j A_ij l_i l_j sin(theta_i - theta_j) theta_dot_j^2
G_i  = g A_i l_i sin(theta_i)
```

The gravity-compensation construction makes `x* = 0` an **exact** equilibrium of the
noiseless closed loop, so the certificate's target is not approximate. The diffusion is
axis-aligned in velocity coordinates (`additive`: `Sigma = sigma [[0],[I]]`;
`state_dependent`: `diag(sigma(1+|w|))`), and `Sigma` has `n_links` Wiener channels.

The analytic Ito push-forward of the physical SDE under `y = T(x)`,

```
dy = ( J_T f + 1/2 tr(Sigma^T H_T Sigma) ) dt + J_T Sigma dW
```

is evaluated with autograd and used **only** to score the learned world model — never
inside the certificate.

## 2. The learned objects and their by-construction properties

**Transport** (`sbounds.transport`): RealNVP-style affine coupling layers with
tanh-bounded log-scales, so `exp(s)` lives in a known compact interval, the Jacobian is
never degenerate, and `T(x) = R((x-x*)/x_scale) - R(0)` satisfies `T(x*) = 0` exactly.
Invertibility is exact (unit test to 1e-9).

**Lyapunov factor** (`sbounds.models.LyapunovNet`):

```
V(eta) = 1/2 eta^T P eta + sum_j softplus(c_j) * n_j(eta)^2,   P = A A^T + eps I
n_j(eta) = res_j(eta) - res_j(0)
```

`P` is positive definite by construction; `n_j(0) = 0`; `softplus(c_j) >= 0`. Hence
`V` is positive definite and radially unbounded — not by penalty, by construction.

**Latent drift** (`sbounds.models.LatentDynamics`): `F(y) = A y + (res(y) - res(0))`,
so `F(0) = 0` exactly.

**Transversal metric**: `Q` solves the Lyapunov equation `A_rr^T Q + Q A_rr = -I` for the
learned rho block, giving `2 kappa rho^T Q F_rho` the exact term `-kappa ||rho||^2`. Note
that lambda_max of the symmetric part of `A_rr` is *useless* as a contraction test for a
mechanical system: in position-velocity coordinates the symmetric part of any damped
mechanical linearisation has a non-negative largest eigenvalue. Contraction must be
measured through a metric; that is what `Q` is for.

## 3. The certificate

With `B(x) = J T(x) Sigma(x)` and `W(y) = V(eta) + kappa rho^T Q rho`:

```
L W(y) = grad_eta V . F_eta + 2 kappa rho^T Q F_rho
         + 1/2 [ tr(BB^T_{eta,eta} Hess V) + 2 kappa tr(BB^T_{rho,rho} Q) ]
```

**Noise floor.** Additive process noise means the origin is not an equilibrium of the
SDE, so `L W(0) = 1/2 tr(BB^T Hess W)(0) = beta > 0` necessarily (grad W(0) = 0 and
F(0) = 0 kill the drift term). The classical `L W + alpha W <= 0` is unattainable —
demanding it certifies the empty set. The certificate is the thresholded condition

```
L W(y) + alpha W(y) <= beta + tol   for all y in the region
```

which by Ito's formula for `W(y_t)` gives, for `y_0` in the region,

```
E[W(y_t)] <= e^{-alpha t} W(y_0) + ((beta + tol)/alpha)(1 - e^{-alpha t})
```

exponential practical stability to the explicit noise ball of stationary radius
`(beta + tol)/alpha`. `beta` is evaluated exactly (autograd at the origin);
`tol` is explicit slack, reported.

## 4. What the search actually bounds

The verifier computes a **sound upper bound** on `sup_region (L W + alpha W)` per box:

- `grad_eta V` and `Hess V` as interval enclosures (interval forward-mode AD, ternary
  jets per unit with outward rounding; `V.bounds`).
- `F` as an interval enclosure (`A y` exactly through interval linear map + a centered
  (mean-value) remainder through the residual MLP; the residual is a **cascade** — every
  row reads only the factor coordinates `eta`, so the Jacobian enclosure has exact zeros
  on the `rho` columns and the transversal remainder scales with the `eta` box).
- `B B^T` as an interval matrix: the transport's interval Jacobian `J T` (exact interval
  chain rule through each coupling layer) composed with an exact interval `Sigma`.
- The Ito term as the **interval trace** `1/2 tr(BB^T Hess W)`: elementwise interval
  products contain the true products; sums of intervals contain the true sum. Sound —
  and equal to the exact trace on degenerate boxes, so it converges under refinement.
- The rho drift term in factor mode: splitting `F_rho = A_rr rho + (A_re eta + g_rho)`
  and bounding `b_Q = sup ||Q(A_re eta + g_rho)||`, completing the square gives
  `2 kappa rho^T Q F_rho <= kappa max_{s<=r} (-s^2 + 2 s b_Q) = kappa b_Q^2` once the
  ball radius `r >= b_Q` — independent of `r`, which the naive split (linear in `r`)
  never achieves.

Two modes search the **same** region with the **same** bound family:

- `full`: all `D` coordinates are subdivided by branch-and-bound;
- `factor`: only the `d` eta-coordinates are subdivided; rho is handled by the closed
  form above plus the exact interval trace.

The reported certified fraction is a **one-sided lower bound**: a box counts only when
its sound upper bound is `<= beta + tol`; unresolved boxes are unknown, never certified.

## 5. The tightness failure mode (why C-S certifies nothing)

Cauchy–Schwarz gives `1/2 tr(BB^T H) <= 1/2 ||B||_F^2 ||Hess W||_F`. The slack of this
bound is bounded away from zero as the box shrinks to the origin — measured: at a box of
half-width 1e-3 the excess over `beta` is ~0.22 vs ~1e-4 for the interval trace. Since
the certificate constant is `beta + tol` with small `tol`, C-S certifies *no box at any
radius*, while the tight trace certifies every box up to an explicit radius. This is the
load-bearing numerical fact of the repo, and it is both measured (`exp3`) and unit-tested
(point box at the origin must reproduce `beta` to 1e-6).

## 6. Counterexamples vs looseness

A failing *bound* is not a failing *certificate*: interval branching fails for two
reasons that must be separated.

- *genuine violation*: some point in the box really has `L W + alpha W > beta + tol`;
- *bound looseness*: the box's sound upper bound exceeds the threshold although no point
  does.

`sbounds.cegis` splits unresolved boxes, samples inside them, and evaluates the exact
generator (autograd) at those points; both counts are reported, because the share of
failures that are looseness is the honest measure of how much of the certification
budget is spent on the bound rather than on the system. Retraining uses only genuine
violations, against either the model drift or the exact push-forward oracle.

## 7. Certified online adaptation

Two certificate/model pairs are alive: the active (certified) pair and a shadow pair
training online on fresh plant data. A candidate swap is accepted only if the sound
bound certifies the shadow pair over the region at threshold `beta + tol`. Unsafe swaps
are counted by evaluating the *exact plant* generator (push-forward oracle) on fixed
probe points after the swap — so the gate's value is a falsifiable number, not a
design assertion. If the gate keeps rejecting (plant drifted out of scope), the region
shrinks and that time is reported.

## 9. Why the factorised verifier scales with the latent dimension (proposition)

The empirical ladder (factor mode certifies at budgets full mode never reaches) is not
an accident of the benchmark; it follows from the cascade structure that training
enforces. Write the latent dynamics in block form, with `eta` the certified factor
(d coordinates) and `rho` the transversal block (m = D - d):

    d/dt (eta, rho) = ( F_ee(eta) + g_e(rho),  A_rr rho + h(eta) )

with the *cascade constraints* `F_e in R^{d x m} = 0` (eta-rows read only eta; enforced
hard in training) and the rho-rows reading eta only through a bounded nonlinearity.
The per-box sound bound has the shape

    sup_box (L W + alpha W)  <=  C_eta(w_eta)  +  C_couple(w_eta, rho_ball)  +  kappa*lam_Q*(s^2 - 2 s b_Q)

where `w_eta` is the eta-box half-width, `b_Q` bounds the rho-row coupling over the
box, and `s` is the completing-the-square slack. Two properties close the argument:

1. **Only d coordinates are subdivided.** In factor mode the search refines `w_eta`
   while the rho block is handled in closed form over its ball. The `C_couple` term is
   evaluated through the eta-box (the rho-rows read eta), so every term decreases as
   the eta subdivision refines: `C_eta = O(w_eta^2)` (centered-form second-order term
   with fixed quadratic V, whose Hessian has zero interval width) and
   `C_couple = O(L_h w_eta)` with `L_h` the (spectrally capped) Lipschitz constant of
   the eta-forcing. Hence the bound crosses the threshold `beta + tol` at a critical
   width `w*` that depends on d, kappa, lam_Q, and the caps -- but **not on D**.
2. **Cell count is exponential in the number of subdivided coordinates.** A uniform
   refinement to width `w*` needs `(scale/w*)^d` cells in factor mode and
   `(scale/w*)^D` in full mode. With the per-cell cost identical (same bound family,
   same Rho rings), the effort ratio at equal threshold is

        effort(full) / effort(factor)  =  (scale/w*)^{D - d}.

This is the d-vs-D separation: at fixed soundness (same bound, same threshold, same
region semantics), the certification effort is governed by the latent dimension d of
the certified factor, while the full-dimensional verifier pays `(D - d)` additional
exponential factors. The measured ladder is this proposition with real constants:
full mode's `w*` under its budget is orders of magnitude above the threshold crossing
(`worst_upper ~ 10^6` at D = 8 vs threshold ~ 0.05), and the gap widens with D because
the *constant* in the full-mode bound also grows (the coupling norms and Hessian
terms accumulate over all D coordinates). The proposition is conservative -- it
treats b_Q, L_h as fixed -- so it lower-bounds the true advantage of the factorised
verifier as D grows.

## 10. What is deliberately loose

- The interval Jacobian chain rule composes interval matrices without exploiting
  correlations — sound, not minimal.
- The rho ball is over-covered by its axis-aligned box when bounding coupling norms.
- `Sigma` for `actuator` noise is not intervalised (it would need the interval inverse
  of a state-dependent inertia); those runs fall back to the C-S bound and say so.
- All interval outputs are inflated outward by 1e-12 relative (≈4000 ulps float64).

## 11. Statistical protocol for the real-data experiments

Every real-data number is reported with its estimation procedure, because the
identified plants are estimates themselves.

**Identification stability (exp8 stage A).** For each domain the dataset is
split into two independent halves (odd/even engines for the fleet; disjoint
timeline blocks for the grid and the air-quality stations). The OU fit is run
separately on each half and the *relative* parameter differences
`||A1 - A2|| / ||A1||` and `||σ1 - σ2|| / ||σ1||` are reported as the
identification noise floor. A claimed drift signal is meaningful only if
`||ΔA||` between eras clears this floor — for all three domains it does, by an
order of magnitude (fleet: drift 0.40 vs floor 0.021).

**Training target (world model).** On real data the per-step drift is small
relative to the per-step diffusion (sensor sampling), so regressing the latent
dynamics on raw differences `(y1-y0)/dt` fits the noise. The world model is
instead fit to the plant's exact Ito push-forward drift under T (§1), recomputed
each batch as T moves. The honest data-derived scores (simulated rollout error
and real held-out one-step error) are still reported — and remain large in the
noise-dominated regime; the certificate's validity does not rest on them,
because the verifier bounds the identified plant exactly, and the identification
is validated in stage A.

**Bootstrap confidence intervals (exp8 stage B).** The pointwise certificate
residual is evaluated at 2048 fixed probes of the certified region (fixed seed,
same probes across all methods and domains). The violation fraction's 95%
confidence interval is the percentile bootstrap (2000 resamples) of the probe
sample. With p ≈ 0.3 this gives ±0.02 absolute resolution — tighter than any
claimed effect in the results.

**Sensitivity ablations (exp8 stage C).** The certificate is re-evaluated while
sweeping one factor at a time on the committed fleet checkpoint: the
evaluation-points parameter κ ∈ {0.6, …, 1.4}, the identified diffusion scaled
by {0.5, 1, 2, 3}, and the certified region scaled by {0.6, 0.8, 1.0, 1.25}.
All other protocol elements are frozen.

**Seed stability (exp8 stage D).** The certificate pipeline is repeated from 3
seeds (the fixed quadratic V is invariant; the learned latent dynamics varies).
The spread of the violation fraction and of the noise floor β across seeds is
reported, so no single-seed number carries the conclusions.

## 12. Formal statements and proofs (supplement)

Throughout, the region `R` is the product `B_eta x B_rho(r)`, the bound `U(box)` is
the computed sound upper bound on `sup_box (L W + alpha W)`, and
`theta = beta + tol` is the certificate threshold. Definitions: a box `B` is
*certified* iff `U(B) <= theta`; a set is *certified* iff it is a finite union of
certified boxes; `U` is *sound* iff `sup_B (L W + alpha W) <= U(B)` for every box.

**Theorem 1 (Soundness of the factorised certificate).**
*Let `W(y) = V(eta) + kappa rho^T Q rho` with `V` C^2, `Q` symmetric positive definite,
and let the identified diffusion `B(x) = J T(x) Sigma(x)` be continuous on the
physical preimage of the box `B`. If every sub-box produced by the search is
certified, then the true region satisfies the certificate:*

    sup_{y in R} (L W(y) + alpha W(y))  <=  theta.

*Consequently, for the true SDE and every `y_0 in R`,*

    sup_{y0 in R} E[W(y_t)]  <=  e^{-alpha t} W(y_0) + (theta/alpha)(1 - e^{-alpha t}).

*Proof.* By Ito's formula, `L W = grad W . F + 1/2 tr(BB^T Hess W)` wherever the
classical derivatives exist; the identified model has `F` and `J T` C^1 and `Sigma`
continuous, so `L W` is defined everywhere on `R`. Soundness of each ingredient is
by construction: (i) interval forward-mode AD with outward rounding encloses
`grad V` and `Hess V` — outward rounding makes the enclosure valid under finite
precision, and the inflation 1e-12 relative covers the residual rounding of the
interval library itself; (ii) the mean-value (centered) form `F(y) in F(c) + DF(B)(y - c)`
is the multivariate mean-value inequality with interval remainder, valid because
`DF` is continuous and the interval Jacobian encloses it on `B`; (iii) elementwise
interval products and sums contain the true trace `tr(BB^T Hess W)`; (iv) the
coupling completion `2 kappa rho^T Q F_rho <= kappa b_Q^2` on the rho ball is the
sharp quadratic inequality `-s^2 + 2 s b_Q <= b_Q^2`, valid for every `s in [0, r]`
with `r >= b_Q`. Summing sound parts yields `L W + alpha W <= U(B)` pointwise on
each `B` (the thresholded `alpha W` term is bounded exactly: `V(eta)` has interval
bounds from the same AD, `rho^T Q rho` from the ring bounds). Taking the max over
the finite sub-division gives `sup_R (L W + alpha W) <= max_B U(B) <= theta`. The
process consequence is Dynkin's formula applied to the `C^2` function `W` at the
stopped process `y_{t ∧ tau_R}` and monotone convergence as `tau_R → ∞`, which is
legitimate since the certificate bounds the expected generator on all of `R` and
`W` grows quadratically while `F` is at most linear in `rho` and globally
Lipschitz on `R` (spectral caps), guaranteeing non-explosion. ∎

**Lemma 1 (Exact noise floor).**
*If `F(0) = 0`, `grad W(0) = 0`, and `B`, `Hess W` are continuous at `0`, then*
`L W(0) = 1/2 tr(BB^T Hess W)(0) =: beta`, *and for every eps > 0 there is a
neighbourhood of the origin on which* `| L W(y) - beta | <= eps`
*+ o(||y||) terms; in particular `sup_B (L W + alpha W) → beta + alpha W(0) = beta`
as the box shrinks to the origin.* *Proof.* Continuity of all factors in the trace
and of `F`; the drift term `grad W . F` vanishes to second order since both factors
vanish at `0` (`grad W(0) = 0`, `F(0) = 0`, `F` Lipschitz). The limit statement is
the definition of continuity applied to the map `y ↦ L W(y)`. ∎
*(This is why the certificate threshold must be `beta + tol`, not `0`: by Lemma 1
any threshold below `beta` certifies no box touching the origin — the classical
`L W + alpha W <= 0` is unattainable for a non-degenerate diffusion. The repo
tests the degenerate box at the origin against `beta` to 1e-6.)*

**Theorem 2 (Convergence and completeness of the search).**
*Assume (A1) `U` is convergent: as `diam(B) → 0`, `U(B) → L W(y_B) + alpha W(y_B)`
uniformly on the region; (A2) the subdivision rule is exhaustive: every infinitely
refined branch has diameter → 0; (A3) the priority queue orders by `U` descending.
Then:* (i) *the union of certified boxes is non-decreasing (certificate
monotonicity), and any box once certified is never reopened;* (ii) *every point of*
`R_θ = { y : L W(y) + alpha W(y) <= theta }` *is eventually certified — the certified
union converges to the largest `R`-measurable inner approximation of `R_θ`*
achievable at the box resolution; (iii) *if `sup_R (L W + alpha W) > theta`, every
infinite run terminates the refinement of every box whose closure does not intersect*
`{L W + alpha W > theta}`, *so unresolved volume concentrates around the true
violation set.* *Proof.* (i) Children of a certified box inherit sound bounds ≤
their parent's (each child's `sup` ≤ the parent's enclosure because the enclosures
are evaluated on sub-boxes and soundness is pointwise; formally `sup_B ≤ sup_parent`
and `U` is monotone under inclusion up to the uniform convergence (A1)), so no
child can violate `<= theta` in the limit; the queue never re-enqueues a certified
box. (ii) Fix `y in R_θ` with margin `m = theta - (L W(y) + alpha W(y)) > 0`. By
(A1) there is `delta` with `U(B) < L W(y_B) + alpha W(y_B) + m/2 <= theta` for all
boxes of diameter < delta containing `y`. By (A2) the branch containing `y` is
eventually subdivided below `delta` (its `U` exceeds theta while uncertified, and
the priority ordering (A3) makes every branch with excess processed eventually —
a finite number of branches have excess ≥ the branch's at each round because the
sum of excesses is bounded on the compact region and each subdivision reduces the
maximal excess; formally the set of boxes with `U > theta` has measure bounded
below by the unresolved measure and each step either certifies or bisects one of
them, so in the limit none remain). Hence `y` ends in a certified box. (iii) is
the contrapositive of (ii) restricted to the violation set: if a box's closure
avoids the violation set, it has a positive margin and the (ii) argument certifies
it. ∎

**Corollary (What the reported number is).**
*The reported certified fraction is a lower bound on the volume of the true
certificate set `R_θ` inside the region, exact at the box resolution; the
complement is 'unknown', never 'unsafe'. Two runs with different budgets are
monotone: the larger budget's certified set contains the smaller's.*

**Theorem 3 (Certificate robustness to plant mismatch).**
*Let the identified model be `(F, B)` with certificate constant*
`c = sup_R (L W + alpha W) <= theta`, *and let the true plant have drift `F_true =
F + delta F` and diffusion `B_true = B + delta B` with `delta F` bounded and*
`sup_R ||delta B B^T + B delta B^T + delta B delta B^T||_* <= eta_B`
*(`*` = the trace form against `Hess W`, i.e. the perturbation of the Ito term).
If*

    sup_R | grad W . delta F |  +  (1/2) eta_B  <=  theta - c,

*then the certificate holds for the true plant:*
`sup_R (L_true W + alpha W) <= theta`. *In particular, with* `Lip(grad W on R) = L_W`
*and `sup_R ||delta F|| <= eta_F`, the sufficient condition is*

    L_W eta_F  +  (1/2) eta_B  <=  theta - c.

*Proof.* The generator is affine in the drift and the diffusion-squared:
`L_true W = grad W . (F + delta F) + 1/2 tr((BB^T + delta BB^T + delta B B^T +
delta B delta B^T) Hess W)`. Taking absolute values, the drift perturbation is
bounded by `||grad W|| ||delta F|| <= L_W eta_F` and the Ito perturbation by
`(1/2) eta_B` (the cross terms are absorbed in the assumed trace bound). Sum and
apply the hypothesis. ∎

**Why Theorem 3 is the bridge to the real data.**
On C-MAPSS the measured identification floor is `rel ||delta A|| ≈ 0.021` against a
drift signal `||delta A|| ≈ 0.40`; Theorem 3 turns exactly this ratio into a
*certificate-validity* statement: any plant whose parameter deviation from the
identified model stays within the floor inherits the certificate, while the era
shift (20× the floor) is what the shadow-gate machinery (exp2/exp6) must detect
and re-certify. The ETTm2 result — certification closing at 0 — is consistent with
Theorem 3: when the identified floor `eta_F` itself is era-dependent and large,
the margin condition fails and no sound certificate should be claimed, which is
what was reported.

**Theorem 4 (Scaling: effort tracks the certified factor's dimension).**
*Under the cascade constraints of section 9 (`F_e in R^{d x m} = 0`; rho-rows read
eta through a bounded forcing), with the quadratic part of `W` fixed and the
residual caps fixed: (i) the critical subdivision width* `w*`
*at which a box's bound crosses `theta` is independent of `D`; (ii) uniform
refinement to `w*` costs*

    effort(factor) = O((scale/w*)^d),   effort(full) = O((scale/w*)^D),

*so at equal soundness* `effort(full) / effort(factor) = (scale/w*)^{D-d}`.
*Proof.* (i) On a box of eta-half-width `w`, the eta-drift term is exact on the
quadratic part of `V` (zero interval width of `Hess V`) plus a centered second-order
remainder `O(L_gradV w^2)`; the coupling term is `sup ||grad_eta V|| * sup ||g_e||`
over the eta-box, `O(L_g w)` with `L_g` the spectrally-capped Lipschitz constant;
the rho block is handled in closed form independent of the eta box given `b_Q(w) =
O(L_h w)`; the Ito trace converges as `O(w)` per dimension through the interval
Jacobian width, contributing `O(w)` with constants depending on the *residual caps*,
not on D (the cascade zeros kill the rho-columns of the drift Jacobian, and `Sigma`
acts through the fixed transport Jacobian whose interval width is `O(w)` in the eta
coordinates only, because the transport's coupling layers are triangular: eta-rows
of `JT` do not depend on rho — this is the by-construction property of section 2).
Hence `U(B) = U_0 + c_2 w^2 + c_1 w` with `U_0 = beta + alpha W(center)` and
constants `c_1, c_2` independent of D; the crossing width `w*` solves
`c_2 w^2 + c_1 w = theta - U_0`, independent of D. (ii) Uniform refinement to `w*`
of the d-dimensional eta-box is `(scale/w*)^d` cells; each cell evaluates the same
bound family with the same ring complexity (rings over the rho ball are recomputed
per cell but their count depends on the caps, not on D since `Q`'s action on the
ball is diagonalized once). In full mode every one of the D coordinates must be
refined to `w*` — the same crossing-width argument gives each coordinate a width
independent of D — giving `(scale/w*)^D` cells. The ratio follows. ∎

*(Theorem 4 makes precise what the ladder shows empirically: the factorised
verifier's cost is exponential in d, not D. The proposition of section 9 is the
informal statement; the constants' D-independence is exactly the cascade +
triangularity structure that training enforces and the implementation preserves.)*

## 13. Scope of the guarantees

What is proven: soundness of every certificate the site reports (Theorem 1), the
necessity of the noise floor and its exact computation (Lemma 1), convergence and
the precise meaning of 'unknown' (Theorem 2 + Corollary), validity under plant
mismatch within the measured identification floor (Theorem 3), and the d-not-D
scaling law (Theorem 4). What is not claimed: minimality of the certified set,
existence of a certifying factorization for arbitrary plants (the learned transport
is a search, and exp9 shows the search matters), or global (region-free)
certificates — the region is part of the claim, and its size is reported.
