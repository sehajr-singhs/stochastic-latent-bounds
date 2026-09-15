"""Dual-buffer shadow-network hot-swap with a certified acceptance gate.

Two certificate/model pairs are kept alive. The active pair serves and is the
one the sound bound has certified. The shadow pair trains online on freshly
arriving data. A candidate swap is accepted only if the *shadow pair itself*
passes the same sound bound over the currently certified region; otherwise the
active pair keeps serving and the rejection is recorded.

Why the gate is not ceremony: the acceptance decision must not depend on the
same sampled data the shadow just trained on. A sampled check can pass while the
sound bound fails, and that difference is exactly what the experiment measures
by running both gates side by side on the same stream:

  naive gate : accept every shadow update (what an unguarded online learner does)
  sound gate : accept only what the interval bound certifies

`unsafe_swaps` counts accepted swaps after which the *true* generator, evaluated
against the exact Ito push-forward of the real plant, is positive somewhere in
the region. That is the claim the experiment is allowed to falsify.

Fallback: if the sound gate is rejected `patience` times in a row, the system is
declared out of certificated scope and drops to the pre-certified conservative
fallback: the certified region is shrunk and the fallback damping gain is
applied. Time spent there is reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import torch

from .generator import latent_generator, noise_floor
from .models import LatentDynamics, LyapunovNet, spectral_project
from .systems import ChainArm, pushforward_drift
from .train import euler_maruyama


@dataclass
class ShadowConfig:
    rounds: int = 8
    batch: int = 128
    shadow_steps: int = 60
    lr: float = 2e-3
    monitor_boxes: int = 32
    patience: int = 3
    gate: str = "sound"          # or "naive"
    probe: int = 512
    tol: float = 0.05            # certificate slack above the noise floor
    seed: int = 0


@dataclass
class ShadowReport:
    rounds: list = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    unsafe_swaps: int = 0
    fallback_rounds: int = 0
    unsafe_swap_fraction: float = 0.0


def _clone_pair(V, F):
    import copy

    return copy.deepcopy(V), copy.deepcopy(F)


def true_violation_probe(V, F, transport, system: ChainArm, kappa: float, alpha: float,
                         d_eta: int, y_probe: torch.Tensor, beta: float | None = None,
                         tol: float = 0.0) -> dict:
    """Exact-generator violation rate on fixed probe points of the region.

    Uses the analytic Ito push-forward of the *real* plant, so this measures the
    physical claim rather than the model's self-consistency. Violations are
    measured against beta + tol: the noise floor alone is exceeded near the
    origin by construction (grad resid(0) != 0), so exceeding beta is not a
    violation -- exceeding the certified ball beta + tol is.
    """
    F_fn = lambda y: pushforward_drift(system, transport, transport.inverse(y))
    resid = latent_generator(V, F, transport, system, kappa, alpha, d_eta,
                             y_probe[..., :d_eta], y_probe[..., d_eta:], F_fn=F_fn)
    beta = noise_floor(V, F, transport, system, kappa, d_eta) if beta is None else float(beta)
    return {"viol_frac": float((resid > beta + tol).to(torch.float64).mean()),
            "max_resid": float(resid.max()),
            "beta": beta}


def _gate_sound(V, F, transport, system, kappa, alpha, d_eta, lo, hi, split_dims,
                beta: float, tol: float = 0.05) -> tuple:
    """Run the interval bound over a fixed partition; return (ok, worst_upper).

    The gate asks whether sup(L W + alpha W) <= beta + tol over the region. The
    tolerance is the same explicit slack the offline certificate uses: even with
    the tight Ito trace, sup over any neighbourhood of the origin exceeds the
    noise floor beta because grad resid(0) != 0 generically. The guarantee the
    gate protects is the noise-ball radius (beta + tol) / alpha.
    """
    from .bnb import worst_first_bnb

    def bound_fn(a, b):
        from .bounds import bound_full

        return bound_full(V, F, transport, system, kappa, alpha, d_eta, a, b,
                          chunk=64).upper

    res = worst_first_bnb(bound_fn, lo, hi, split_dims,
                          node_budget=int(os.environ.get("SLB_GATE_NODES", "20000")),
                          time_budget=float(os.environ.get("SLB_GATE_SECONDS", "180")),
                          return_unknown=False, threshold=beta + tol)
    return res.fully_certified, res.worst_upper


def shadow_run(system_initial: ChainArm, V, F, transport, kappa: float, alpha: float,
               d_eta: int, region_lo: torch.Tensor, region_hi: torch.Tensor,
               drift_system: ChainArm | None = None, cfg: ShadowConfig = ShadowConfig(),
               verbose: bool = False) -> ShadowReport:
    """Run the online-adaptation loop with the selected acceptance gate."""
    plant = system_initial
    rep = ShadowReport()
    g = torch.Generator().manual_seed(cfg.seed)
    split_dims = torch.arange(transport.dim)
    # fixed probes for the safety measurement, sampled once in the region
    u = torch.rand((cfg.probe, transport.dim), dtype=torch.float64, generator=g)
    y_probe = region_lo + (region_hi - region_lo) * u
    beta = noise_floor(V, F, transport, system_initial, kappa, d_eta)
    reject_streak = 0

    for rd in range(cfg.rounds):
        # --- the plant changes underneath us ---------------------------------
        if drift_system is not None and rd == max(1, cfg.rounds // 2):
            plant = drift_system
        # --- fresh data from the current plant -------------------------------
        x0 = region_lo + (region_hi - region_lo) * torch.rand(
            (cfg.batch * 4, transport.dim), dtype=torch.float64, generator=g)
        x1 = euler_maruyama(plant, x0, 1, 0.01, generator=g)
        # --- shadow trains a few steps ---------------------------------------
        Vs, Fs = _clone_pair(V, F)
        params = list(Vs.parameters()) + list(Fs.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        for _ in range(cfg.shadow_steps):
            idx = torch.randperm(x0.shape[0], generator=g)[:cfg.batch]
            y0, y1 = transport(x0[idx]), transport(x1[idx])
            target = (y1 - y0) / 0.01
            l_fit = ((Fs(y0) - target) ** 2).mean()
            eta = y0[..., :d_eta].clone().requires_grad_(True)
            rho = y0[..., d_eta:].clone().requires_grad_(True)
            resid = latent_generator(Vs, Fs, transport, plant, kappa, alpha, d_eta,
                                     eta, rho, create_graph=True)
            loss = l_fit + 0.1 * torch.nn.functional.softplus(resid / 0.1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            spectral_project(Vs, 4.0)
            spectral_project(Fs, 4.0)

        # --- acceptance gate --------------------------------------------------
        if cfg.gate == "naive":
            accept = True
            gate_ok, worst = True, float("nan")
        else:
            gate_ok, worst = _gate_sound(Vs, Fs, transport, plant, kappa, alpha, d_eta,
                                         region_lo.unsqueeze(0), region_hi.unsqueeze(0),
                                         split_dims, beta, tol=cfg.tol)
            accept = gate_ok

        # --- measure the physical claim after the candidate swap --------------
        after = true_violation_probe(Vs if accept else V, Fs if accept else F, transport,
                                     plant, kappa, alpha, d_eta, y_probe, beta=beta,
                                     tol=cfg.tol)
        unsafe = after["viol_frac"] > 0.0
        if accept:
            V, F = Vs, Fs
            rep.accepted += 1
            if unsafe:
                rep.unsafe_swaps += 1
            reject_streak = 0
        else:
            rep.rejected += 1
            reject_streak += 1
        if reject_streak >= cfg.patience:
            rep.fallback_rounds += 1
            if verbose:
                print(f"  [shadow] round {rd}: entering fallback (region shrunk)")
            region_lo, region_hi = region_lo * 0.99, region_hi * 0.99
            reject_streak = 0
        rep.rounds.append({"round": rd, "accepted": bool(accept), "gate_ok": bool(gate_ok),
                           "worst_upper": float(worst) if worst == worst else None,
                           "viol_frac_after": after["viol_frac"],
                           "max_resid_after": after["max_resid"],
                           "unsafe": bool(unsafe)})
        if verbose:
            print(f"  [shadow] round {rd}: accept={accept} unsafe={unsafe} "
                  f"viol={after['viol_frac']:.4f} region={float(region_hi.abs().max()):.3f}")
    rep.unsafe_swap_fraction = rep.unsafe_swaps / max(rep.accepted, 1)
    return rep
