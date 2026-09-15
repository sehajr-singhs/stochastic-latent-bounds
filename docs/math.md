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
- `F` as an interval enclosure (`A y` exactly through interval linear map + IBP through
  the residual MLP).
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

## 8. What is deliberately loose

- The interval Jacobian chain rule composes interval matrices without exploiting
  correlations — sound, not minimal.
- The rho ball is over-covered by its axis-aligned box when bounding coupling norms.
- `Sigma` for `actuator` noise is not intervalised (it would need the interval inverse
  of a state-dependent inertia); those runs fall back to the C-S bound and say so.
- All interval outputs are inflated outward by 1e-12 relative (≈4000 ulps float64).
