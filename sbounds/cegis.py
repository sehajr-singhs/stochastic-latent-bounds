"""Counterexample search and counterexample-guided retraining (CEGIS).

The central honesty point of this module is that a failing *bound* is not the
same as a failing *certificate*. Interval branching fails for two different
reasons and they must not be conflated:

  genuine violation   a point inside the box really has L W + alpha W > 0
  bound looseness     the box's sound upper bound is positive although no point
                      in it violates the condition

Only the first is a counterexample worth retraining on. This module separates
them by splitting the boxes the verifier could not certify, sampling inside
them, and evaluating the exact generator. The count of each is reported, because
the share of failures that are looseness is the honest measure of how much of
the certification budget is being spent on the bound rather than on the system.

Two oracle choices are exposed and both are reported:

  model   : the learned latent drift F, i.e. the certificate of the model
  exact   : the analytic Ito push-forward of the real plant, i.e. the physical
            claim. Disagreement between the two is the model-error margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .bnb import BnBResult, box_measure, worst_first_bnb
from .generator import latent_generator, noise_floor
from .train import CertConfig, train_certificate


def _beta(V, F, transport, system, kappa, d_eta, beta) -> float:
    """Resolve the certificate threshold: explicit value or the exact noise floor."""
    return noise_floor(V, F, transport, system, kappa, d_eta) if beta is None else float(beta)


#: Default slack above the noise floor. sup(L W + alpha W) exceeds the floor
#: beta on every neighbourhood of the origin because grad resid(0) != 0
#: generically, so comparing against bare beta would report permanent "genuine
#: violations" that are really the noise floor doing its job. The certificate
#: is L W + alpha W <= beta + tol, i.e. a noise ball of radius (beta+tol)/alpha.
DEFAULT_TOL = 0.05


@torch.no_grad()
def random_violation_rate(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
                          eta_scale: torch.Tensor, rho_radius: float, n: int = 20000,
                          seed: int = 0, F_fn=None, beta: float | None = None,
                          tol: float = DEFAULT_TOL) -> dict:
    """Violation rate under uniform sampling of the eta-box, uniform in the rho ball."""
    g = torch.Generator().manual_seed(seed)
    m = len(rho_radius) if isinstance(rho_radius, torch.Tensor) else transport.dim - d_eta
    r = float(rho_radius) if not isinstance(rho_radius, torch.Tensor) else float(rho_radius.max())
    u = torch.rand((n, d_eta), dtype=torch.float64, generator=g)
    eta = (2.0 * u - 1.0) * eta_scale
    # uniform in the ball by radial scaling with the correct exponent
    v = torch.randn((n, m), dtype=torch.float64, generator=g)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rad = r * torch.rand((n, 1), dtype=torch.float64, generator=g) ** (1.0 / m)
    rho = v * rad
    resid = latent_generator(V, F, transport, system, kappa, alpha, d_eta, eta, rho,
                             F_fn=F_fn)
    beta = _beta(V, F, transport, system, kappa, d_eta, beta)
    return {"viol_frac": float((resid > beta + tol).to(torch.float64).mean()),
            "max_resid": float(resid.max()),
            "beta": beta,
            "tol": tol,
            "q999_resid": float(resid.quantile(0.999))}


def pgd_counterexamples(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
                        eta_scale: torch.Tensor, rho_radius: float, n_starts: int = 256,
                        steps: int = 60, lr: float = 0.05, seed: int = 0,
                        F_fn=None, beta: float | None = None,
                        tol: float = DEFAULT_TOL) -> dict:
    """Maximise L W + alpha W inside the region by projected gradient ascent."""
    g = torch.Generator().manual_seed(seed)
    m = transport.dim - d_eta
    r = float(rho_radius)
    u = torch.rand((n_starts, d_eta), dtype=torch.float64, generator=g)
    eta = ((2.0 * u - 1.0) * eta_scale).clone()
    v = torch.randn((n_starts, m), dtype=torch.float64, generator=g)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rho = (v * r * torch.rand((n_starts, 1), dtype=torch.float64, generator=g)).clone()

    lam = torch.tensor(1.0, dtype=torch.float64)
    for it in range(steps):
        eta_r = eta.clone().requires_grad_(True)
        rho_r = rho.clone().requires_grad_(True)
        resid, grad = latent_generator(V, F, transport, system, kappa, alpha, d_eta,
                                       eta_r, rho_r, F_fn=F_fn, create_graph=True,
                                       need_grad=True)
        with torch.no_grad():
            eta = eta + lam * lr * grad[..., :d_eta]
            rho = rho + lam * lr * grad[..., d_eta:]
            eta = eta.clamp(-eta_scale.abs(), eta_scale.abs())
            nr = rho.norm(dim=-1, keepdim=True)
            rho = torch.where(nr > r, rho * (r / nr.clamp_min(1e-12)), rho)
            lam = lam * 0.99
    with torch.no_grad():
        resid = latent_generator(V, F, transport, system, kappa, alpha, d_eta, eta, rho,
                                 F_fn=F_fn)
        beta = _beta(V, F, transport, system, kappa, d_eta, beta)
        viol = resid > beta + tol
    return {"viol_frac": float(viol.to(torch.float64).mean()),
            "max_resid": float(resid.max()),
            "beta": beta,
            "tol": tol,
            "n_violating": int(viol.sum()),
            "eta_violating": eta[viol].detach(),
            "rho_violating": rho[viol].detach(),
            "resid": resid.detach()}


@dataclass
class CegisRound:
    round: int
    certified_fraction: float
    nodes: int
    unknown_boxes: int
    genuine_violations: int
    looseness_failures: int
    max_resid_sampled: float
    max_resid_pgd: float
    seconds: float


@dataclass
class CegisReport:
    rounds: list = field(default_factory=list)
    final_certified_fraction: float = 0.0
    total_unknown_examined: int = 0
    total_genuine: int = 0
    total_looseness: int = 0


def cegis_loop(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
               bound_fn_factory, lo0: torch.Tensor, hi0: torch.Tensor,
               split_dims: torch.Tensor,               node_budget: int = 4000, rounds: int = 3,
               samples_per_box: int = 8, retrain_cfg: CertConfig | None = None,
               oracle: str = "model", beta: float | None = None,
               tol: float = DEFAULT_TOL, verbose: bool = False) -> CegisReport:
    """Verifier-in-the-loop retraining. `oracle` selects model or exact drift.

    beta is the certificate constant: a box passes when its sound bound <=
    beta + tol and a point is a genuine violation when the exact generator
    exceeds beta + tol. With additive process noise beta is the noise floor
    L W(0); the slack tol is required because sup exceeds the floor near the
    origin no matter how good the certificate is.
    """
    rep = CegisReport()
    beta = _beta(V, F, transport, system, kappa, d_eta, beta)
    F_fn = None
    if oracle == "exact":
        from .systems import pushforward_drift

        def F_fn(y):
            return pushforward_drift(system, transport, transport.inverse(y))

    for r in range(rounds):
        res: BnBResult = worst_first_bnb(bound_fn_factory(1.0), lo0, hi0, split_dims,
                                         node_budget=node_budget, return_unknown=True,
                                         threshold=beta + tol)
        n_unknown = 0 if res.unknown_boxes is None else res.unknown_boxes.shape[0]
        genuine, looseness = 0, 0
        max_sampled = float("-inf")
        viol_eta, viol_rho = [], []
        if res.unknown_boxes is not None:
            g = torch.Generator().manual_seed(1000 + r)
            for k in range(res.unknown_boxes.shape[0]):
                lo, hi = res.unknown_boxes[k], res.unknown_boxes_hi[k]
                u = torch.rand((samples_per_box, transport.dim), dtype=torch.float64,
                               generator=g)
                y = lo + (hi - lo) * u
                resid = latent_generator(V, F, transport, system, kappa, alpha, d_eta,
                                         y[..., :d_eta], y[..., d_eta:], F_fn=F_fn)
                max_sampled = max(max_sampled, float(resid.max()))
                bad = resid > beta + tol
                if bool(bad.any()):
                    genuine += 1
                    viol_eta.append(y[bad][:, :d_eta])
                    viol_rho.append(y[bad][:, d_eta:])
                else:
                    looseness += 1
        pgd = None
        if r < rounds - 1:
            scale = lo0[..., :d_eta].abs()
            pgd = pgd_counterexamples(V, F, transport, system, kappa, alpha, d_eta,
                                      scale, float(lo0[..., d_eta:].abs().max()),
                                      n_starts=128, steps=40, seed=r, F_fn=F_fn)
            genuine += pgd["n_violating"]
            if pgd["n_violating"]:
                viol_eta.append(pgd["eta_violating"])
                viol_rho.append(pgd["rho_violating"])
        rep.rounds.append(CegisRound(
            round=r, certified_fraction=res.certified_fraction, nodes=res.nodes,
            unknown_boxes=n_unknown, genuine_violations=genuine,
            looseness_failures=looseness,
            max_resid_sampled=max_sampled if max_sampled > float("-inf") else float("nan"),
            max_resid_pgd=pgd["max_resid"] if pgd else float("nan"),
            seconds=res.seconds))
        rep.total_unknown_examined += n_unknown
        rep.total_genuine += genuine
        rep.total_looseness += looseness
        rep.final_certified_fraction = res.certified_fraction
        if verbose:
            rr = rep.rounds[-1]
            print(f"  [cegis] round {r}: cert {rr.certified_fraction:.4f} nodes {rr.nodes} "
                  f"unknown {n_unknown} genuine {genuine} looseness {looseness}")
        if genuine == 0:
            if verbose:
                print("  [cegis] no genuine violations remain; stopping")
            break
        ex_eta = torch.cat(viol_eta, dim=0) if viol_eta else None
        ex_rho = torch.cat(viol_rho, dim=0) if viol_rho else None
        cfg = retrain_cfg or CertConfig(steps=400)
        train_certificate(V, F, transport, system, d_eta, cfg,
                          extra_eta=ex_eta, extra_rho=ex_rho, extra_weight=50.0,
                          use_exact_drift=(oracle == "exact"), verbose=False)
    return rep
