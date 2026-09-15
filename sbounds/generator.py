"""Exact evaluation of the latent stochastic generator L W + alpha W.

For W(y) = V(eta) + kappa rho^T Q rho on the transported coordinates y = T(x) of
the physical SDE dx = f dt + Sigma dW, with B = J T(x) Sigma(x):

    L W(y) = grad W . F(y) + 1/2 tr( B B^T Hess_y W )
           = grad_eta V . F_eta + 2 kappa rho^T Q F_rho
             + 1/2 [ tr(BB^T_{eta,eta} Hess V) + 2 kappa tr(BB^T_{rho,rho} Q) ]

Q is not free. It solves the Lyapunov equation A_rr^T Q + Q A_rr = -I for the
learned rho block, which makes rho^T Q F_rho carry the exact term -||rho||^2 and
turns transversal contraction into a solvable linear system instead of a search.
The naive test - lambda_max of the symmetric part of A_rr - is useless here: the
linearisation of any damped mechanical system is non-normal in position-velocity
coordinates, so its symmetric part has a non-negative eigenvalue regardless of
how strongly it contracts. `model.refresh_rho_metric` reports both.

Two structural facts keep this cheap, and both come from W being quadratic in rho:

  * the Ito trace needs only Hess V on the d-dimensional factor,
  * the rho block contributes a diagonal read-off of BB^T.

So the expensive part of the generator costs O(d) second derivatives, not O(D).

`F_fn` lets the caller swap the learned drift for the exact Ito push-forward
(`systems.pushforward_drift`), which is how the certificate is scored against the
real plant rather than only against its model.

**The noise floor.** With additive process noise the origin is not an equilibrium
of the SDE: paths leave it immediately, so L W(0) = 1/2 tr(B B^T Hess W)(0) > 0
and the classical condition L W + alpha W <= 0 is unattainable -- demanding it
certifies the empty set, always. The honest certificate is the thresholded one,

    L W + alpha W <= beta,   beta := L W(0),

which by Ito's formula gives E W(t) <= e^{-alpha t} W(0) + (beta/alpha)(1-e^{-alpha t}):
exponential practical stability to an explicit noise ball of stationary radius
beta/alpha. `noise_floor` evaluates beta exactly at the origin; every verifier in
this package compares against it.
"""
from __future__ import annotations

import torch


def noise_floor(V, F, transport, system, kappa: float, d_eta: int) -> float:
    """The exact generator at the origin: beta = L W(0) = 1/2 tr(BB^T Hess W)(0).

    grad W(0) = 0 (W has its minimum there) and F(0) = 0, so the drift term
    vanishes and beta is purely the Ito term. alpha does not enter: W(0) = 0.
    """
    D = transport.dim
    zero = torch.zeros(1, D, dtype=torch.float64)
    beta = latent_generator(V, F, transport, system, kappa, 0.0, d_eta,
                            zero[..., :d_eta], zero[..., d_eta:])
    return float(beta[0])


def latent_generator(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
                     eta: torch.Tensor, rho: torch.Tensor, F_fn=None,
                     create_graph: bool = False, need_grad: bool = False):
    """Evaluate L W + alpha W at the given latent points.

    Returns resid (B,) and, when need_grad, the gradient of resid w.r.t.
    the concatenated (eta, rho).
    """
    Q = F.rho_metric(d_eta)
    eta = eta.clone().requires_grad_(True)
    rho = rho.clone().requires_grad_(True)
    y = torch.cat([eta, rho], dim=-1)
    D = y.shape[-1]

    W = V(eta) + kappa * ((rho @ Q) * rho).sum(-1)
    gV = torch.autograd.grad(W.sum(), eta, create_graph=True)[0]           # (B,d)
    gW = torch.cat([gV, 2.0 * kappa * (rho @ Q)], dim=-1)                  # (B,D)

    Fv = F(y) if F_fn is None else F_fn(y)
    drift = (gW * Fv).sum(-1)

    x = transport.inverse(y)
    J = torch.stack([torch.autograd.grad(transport(x)[..., k].sum(), x,
                                         create_graph=True, retain_graph=True)[0]
                     for k in range(D)], dim=-2)                           # (B,D,D)
    sig = system.diffusion(x)
    Bm = J @ sig
    BBt = Bm @ Bm.transpose(-1, -2)                                        # (B,D,D)

    hess_v = [torch.autograd.grad(gV[:, i].sum(), eta, create_graph=True, retain_graph=True)[0]
              for i in range(d_eta)]
    Hv = torch.stack(hess_v, dim=-2)                                       # (B,d,d)
    term_eta = (BBt[:, :d_eta, :d_eta] * Hv).sum(dim=(-1, -2))
    term_rho = 2.0 * kappa * (BBt[:, d_eta:, d_eta:] * Q).sum(dim=(-1, -2))
    ito = 0.5 * (term_eta + term_rho)

    resid = drift + ito + alpha * W
    if not need_grad:
        return resid if create_graph else resid.detach()
    g = torch.autograd.grad(resid.sum(), [eta, rho], create_graph=create_graph)
    return resid, torch.cat([g[0], g[1]], dim=-1)
