# stochastic-latent-bounds

**Sound stochastic Lyapunov certificates for high-dimensional physical systems, via invertible latent transports.**

A trained world model plus a learned diffeomorphism turn a stochastic physical system into
coordinates where a Lyapunov certificate is *searchable*: the certificate is verified by
interval branch-and-bound with **sound bounds** (outward-rounded interval arithmetic over
value, gradient and Hessian enclosures of the learned maps), never assumed from sampling.

Live site: **https://sehajr-singhs.github.io/stochastic-latent-bounds/**
(built with the same `pages-themes/minimal` look as [e3nn.org](https://e3nn.org) — `/docs` folder, GitHub Pages).

---

## The certificate, precisely

For the physical SDE `dx = f(x) dt + Sigma(x) dW` and an exactly invertible transport
`y = T(x)` with `T(x*) = 0`, factoring `y = (eta, rho)` into a `d`-dimensional certified
factor and a transversal residual, define

```
W(y) = V(eta) + kappa rho^T Q rho          (Q: A_rr^T Q + Q A_rr = -I)
L W(y) = grad W . F(y) + 1/2 tr( B B^T Hess W ),   B = J T(x) Sigma(x)
```

**The noise floor.** With additive process noise the origin is *not* an equilibrium of the
SDE: paths leave it immediately, so `L W(0) = beta > 0` and the classical condition
`L W + alpha W <= 0` certifies the empty set, always. The attainable statement is

```
L W + alpha W <= beta + tol   for all y in the region
  =>  E[W(t)] <= e^{-alpha t} W(0) + ((beta+tol)/alpha)(1 - e^{-alpha t})
```

exponential practical stability to an explicit noise ball of stationary radius
`(beta + tol)/alpha`. `beta = L W(0)` is evaluated exactly (autograd); `tol` is explicit
slack, reported rather than hidden.

**Why two bound constructions matter.** The usual sound enclosures are too loose to certify
anything at region scale. First, the Cauchy–Schwarz Ito enclosure
`1/2 ||B||_F^2 ||Hess W||_F` has slack that does **not** vanish as boxes shrink; near the
origin it exceeds `beta` by orders of magnitude and no box certifies at any radius. This repo
instead encloses `BB^T` itself as an interval matrix (interval Jacobian of the transport
composed with an exact interval `Sigma`) and contracts it against the interval Hessian of
`W` — sound, and exact on degenerate boxes (`ito_mode="tight"` vs `"cs"`, measured in
`experiments/exp3_tightness.py`). Second, raw interval bound propagation of the drift terms
is *linear* in box width and its slack exceeds the threshold even after millions of BnB
nodes; the verifier therefore uses **centered (mean-value) forms**: the residual is evaluated
exactly at the box center and the deviation bounded through a Jacobian enclosure whose
entries are exact zeros (the factorised residual's eta-rows do not read rho — enforced as a
hard projection during training) or spectral-norm Lipschitz balls. The remainder is
*quadratic* in box radius, which is what makes branch-and-bound converge. Both claims are
unit-tested.

## Verified claims (tested, not promised)

- **Soundness** — `tests/test_soundness.py`: across random boxes the sound bound must
  dominate the exact generator at every sampled interior point; any box the verifier
  declares certified must satisfy the certificate at every sampled point. Unsoundness here
  is the one bug this repo must not have.
- **Tightness** — a point box at the origin must reproduce `beta` to 1e-6. The C-S bound
  structurally cannot pass this.
- **Invertibility** — `T^{-1}(T(x)) = x` to 1e-9; `T(x*) = 0` exactly.
- **World model honesty** — the latent model is trained on rollout pairs only (finite
  differences of pushed-forward observations) and scored against the analytic Ito
  push-forward of the real plant, including the `1/2 tr(Sigma^T H_T Sigma)` correction term
  most latent models silently absorb.
- **Certified online adaptation** — shadow networks hot-swap behind a certified acceptance
  gate; unsafe swaps are counted against the *exact* plant generator, so the gate's value is
  falsifiable.

## What is *not* claimed

- No global ("all of R^D") certificate: the claim is over an explicit region (90th
  percentile of visited states), sized from data.
- The dimension reduction is a **factorisation**, not a fictitious R^D -> R^d bijection: a
  diffeomorphism of R^D cannot drop dimensions; the residual block carries a proven
  transversal contraction and an explicit penalty in `W`.
- The certificate bounds the **model's** latent dynamics by construction; agreement with the
  physical plant is measured (model-error margin) via the exact push-forward oracle, not
  assumed. Where the two disagree, the number is reported.

## Quick start

```bash
pip install torch pytest

# soundness + tightness suite (CPU, float64)
python -m pytest tests/ -q

# experiments (checkpointed; re-runs skip finished stages)
python experiments/exp1_main.py          # factor vs full certified-volume scaling
python experiments/exp2_shadow.py        # gated hot-swap under plant drift
python experiments/exp3_tightness.py     # tight trace vs Cauchy-Schwarz

# figures + results page (pure Python, no matplotlib)
python tools/figures.py
python tools/site.py
```

All heavy runs were executed on a dedicated Linux box (8 cores, CPU torch, float64);
`results/*.json` are committed next to the code that produced them.

## Repo layout

```
sbounds/            the framework
  systems.py        N-link arm, analytic drift/diffusion, exact Ito push-forward
  transport.py      invertible coupling transport + interval Jacobians
  models.py         LyapunovNet (PD by construction), LatentDynamics (F(0)=0)
  nets.py           interval arithmetic + interval forward-mode AD (jets)
  bounds.py         sound bounds: tight interval Ito trace, C-S baseline
  region.py         regions + certification entry points (beta + tol threshold)
  bnb.py            worst-first branch-and-bound with node/time budgets
  generator.py      exact latent generator + noise floor (evaluation only)
  train.py          data gen, world-model training, certificate training
  cegis.py          counterexample search: genuine violations vs bound looseness
  shadow.py         dual-buffer hot-swap behind a certified gate
experiments/        exp1 scaling, exp2 shadow, exp3 tightness (checkpointed)
tests/              soundness / tightness / invertibility / metric tests
tools/              figure + site generators (pure Python SVG)
docs/               GitHub Pages site (e3nn-style minimal look)
results/            committed JSON outputs of every experiment
```

## Site

The GitHub Pages site in `docs/` renders every number from `results/*.json` via
`tools/site.py`, so the published page cannot drift from what the code produced.
Regenerate and commit `docs/` together with `results/`.
