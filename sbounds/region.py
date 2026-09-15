"""Region definitions and bound-function factories.

A region is the set on which the certificate is claimed, expressed in the
transported coordinates y:

    R(scale) = { eta : |eta_i| <= scale * eta_scale_i }  x  { ||rho|| <= scale * rho_radius }

Both certification modes verify this same set. The only difference is what BnB
is allowed to subdivide, which is what makes the cost comparison meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .bnb import bisect_radius, worst_first_bnb
from .bounds import bound_factor, bound_full


@dataclass
class Region:
    d_eta: int
    eta_scale: torch.Tensor        # (d,)
    rho_radius: float
    full_scale: torch.Tensor       # (D,) used by the full mode

    @staticmethod
    def from_transport(transport, d_eta: int, rho_radius: float, eta_scale: float = 0.5,
                       full_scale: float = 0.5) -> "Region":
        return Region(
            d_eta=d_eta,
            eta_scale=torch.full((d_eta,), eta_scale, dtype=torch.float64),
            rho_radius=rho_radius,
            full_scale=torch.full((transport.dim,), full_scale, dtype=torch.float64),
        )

    def init_box(self, mode: str, scale: float = 1.0):
        if mode == "factor":
            s = self.eta_scale * scale
            return -s.unsqueeze(0), s.unsqueeze(0)
        s = self.full_scale * scale
        return -s.unsqueeze(0), s.unsqueeze(0)

    def split_dims(self, mode: str) -> torch.Tensor:
        if mode == "factor":
            return torch.arange(self.d_eta)
        return torch.arange(len(self.full_scale))


def make_bound_fn(mode: str, V, F, transport, system, kappa: float, alpha: float,
                  region: Region, rho_radius: float | None = None,
                  chunk: int = 96, scale: float = 1.0, ito_mode: str = "tight",
                  rho_rings: int = 8):
    """Return bound_fn(lo, hi) -> sound upper bounds for the chosen mode.

    In factor mode the rho ball is partitioned into `rho_rings` shells and the
    bound of the worst shell is returned. This is sound (the ball is the union
    of the shells) and necessary: on the full ball the completing-the-square
    term and the alpha*W ball term do not shrink with the eta box, so no
    eta-box could ever certify. On inner shells both shrink with the shell
    radius, which is what makes the factorised search converge.
    """
    d = region.d_eta
    r = region.rho_radius if rho_radius is None else rho_radius

    if mode == "factor":
        def bf(lo, hi):
            worst = None
            for k in range(rho_rings):
                r_out = r * (k + 1) / rho_rings
                r_in = r * k / rho_rings
                res = bound_factor(V, F, transport, system, kappa, alpha, d, lo, hi,
                                   r_out, chunk=chunk, ito_mode=ito_mode,
                                   rho_rings=1, r_in=r_in).upper
                worst = res if worst is None else torch.maximum(worst, res)
            return worst
    else:
        def bf(lo, hi):
            return bound_full(V, F, transport, system, kappa, alpha, d, lo, hi,
                              chunk=chunk, ito_mode=ito_mode).upper
    return bf


def certify_region(mode: str, V, F, transport, system, kappa: float, alpha: float,
                   region: Region, node_budget: int = 4000, chunk: int = 96,
                   time_budget: float | None = 300.0, scale: float = 1.0,
                   threshold: float | None = None, beta: float | None = None,
                   tol: float = 0.05, ito_mode: str = "tight") -> dict:
    """Run BnB on the region and return certified fraction plus cost.

    threshold is the certificate constant: a box is certified when
    sup(L W + alpha W) <= threshold. With additive process noise the attainable
    constant is the noise floor beta = L W(0) > 0, and even with the tight Ito
    trace sup over any neighbourhood of the origin exceeds beta, because
    grad resid(0) != 0 generically. The threshold is therefore beta + tol with
    tol an explicitly reported slack: the guarantee becomes E[W] bounded to the
    noise ball of stationary radius (beta + tol) / alpha.
    """
    from .generator import noise_floor

    if threshold is None:
        b = noise_floor(V, F, transport, system, kappa, d_eta=region.d_eta) if beta is None else float(beta)
        threshold = b + tol
    lo, hi = region.init_box(mode, scale)
    bf = make_bound_fn(mode, V, F, transport, system, kappa, alpha, region,
                       chunk=chunk, ito_mode=ito_mode)
    res = worst_first_bnb(bf, lo, hi, region.split_dims(mode), node_budget=node_budget,
                          time_budget=time_budget, return_unknown=False,
                          threshold=threshold)
    return {"mode": mode, "scale": scale, "certified_fraction": res.certified_fraction,
            "nodes": res.nodes, "seconds": res.seconds, "worst_upper": res.worst_upper,
            "fully_certified": res.fully_certified}


def certified_radius(mode: str, V, F, transport, system, kappa: float, alpha: float,
                     region: Region, node_budget: int = 2000, chunk: int = 96,
                     quantile: float = 0.99, iters: int = 5,
                     time_budget: float | None = 240.0, threshold: float | None = None,
                     beta: float | None = None, tol: float = 0.05,
                     ito_mode: str = "tight") -> dict:
    """Largest radial scale whose region certifies to `quantile` under the budget."""
    from .generator import noise_floor

    if threshold is None:
        b = noise_floor(V, F, transport, system, kappa, d_eta=region.d_eta) if beta is None else float(beta)
        threshold = b + tol

    def factory(scale):
        return make_bound_fn(mode, V, F, transport, system, kappa, alpha, region,
                             chunk=chunk, scale=scale, ito_mode=ito_mode)

    lo, hi = region.init_box(mode, 1.0)
    out = bisect_radius(factory, lo, hi, region.split_dims(mode), node_budget=node_budget,
                        quantile=quantile, iters=iters, time_budget=time_budget,
                        threshold=threshold)
    out["mode"] = mode
    return out
