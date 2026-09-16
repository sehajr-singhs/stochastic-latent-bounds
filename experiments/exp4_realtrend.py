"""Experiment 4: the framework on real fleet data (NASA C-MAPSS FD001).

Pipeline, end to end on real data:

1. IDENTIFY. The 21 sensor channels of the 100-engine FD001 split are reduced
   to 8 operational channels and z-scored (stats from fit units only). A
   linear-Gaussian (Ornstein-Uhlenbeck) model is fit by least squares on
   within-engine first differences, separately for the healthy era (first half
   of each engine's life) and the aged era (second half). The identified drift
   difference ||dA|| between eras is the real, measured distribution shift.

2. TRAIN. The invertible transport + latent world model are trained on rollouts
   of the healthy identified system -- the same pipeline as exp1, with the
   simulated N-link arm replaced by a model whose parameters come from NASA
   trajectories.

3. CERTIFY. Factor-mode (d=2 latent coordinates) vs full-mode (all D=8) branch
   and bound at the same node budget, threshold beta + tol with beta the exact
   noise floor of the identified diffusion.

4. ERA SEPARATION. The physical premise of the gate, measured on the identified
   plants: the fraction of region probes where the certificate's generator
   condition is violated under the aged plant vs the healthy plant.

5. GATED HOT-SWAP. The exp2 shadow protocol with the drift event being the
   REAL aged-era plant (not a synthetic damping change).

Everything is checkpointed; results land in results/exp4_realtrend.json.

Usage:  python experiments/exp4_realtrend.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, _jsonable
from experiments.exp1_main import certify_both
from sbounds.bnb import worst_first_bnb
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region, make_bound_fn
from sbounds.realsys import era_systems, load_cmapss_fd001
from sbounds.shadow import ShadowConfig, shadow_run, true_violation_probe
from sbounds.train import (CertConfig, RolloutData, TrainConfig,
                           euler_maruyama, generate_pairs, score_world_model,
                           train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

CKPT = os.path.join(RESULTS, "exp4_ckpt.pt")
SPLIT = 0.5            # healthy = first SPLIT of each engine's life
D_ETA = 2
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
RHO_RINGS = 8


def _era_masks(unit: torch.Tensor, split: float):
    lo = torch.zeros_like(unit, dtype=torch.bool)
    hi = torch.zeros_like(unit, dtype=torch.bool)
    for u in unit.unique():
        idx = torch.nonzero(unit == u).squeeze(-1)
        n = idx.numel()
        cut = int(n * split)
        lo[idx[:cut]] = True
        hi[idx[cut:]] = True
    return lo, hi


def era_probe(trained: dict, sys_healthy, sys_aged, n_probe: int = 512) -> dict:
    """Certificate-violation fraction under each era's plant, same probes."""
    V, F, transport = trained["V"], trained["F"], trained["transport"]
    region, d = trained["region"], trained["d_eta"]
    lo, hi = region.init_box("full")
    g = torch.Generator().manual_seed(7)
    u = torch.rand((n_probe, transport.dim), dtype=torch.float64, generator=g)
    y_probe = lo + (hi - lo) * u
    out = {}
    for name, plant in (("healthy", sys_healthy), ("aged", sys_aged)):
        pr = true_violation_probe(V, F, transport, plant, KAPPA, ALPHA, d,
                                  y_probe, beta=trained["beta"], tol=TOL)
        out[name] = {"viol_frac": pr["viol_frac"], "mean_resid": pr.get("mean_resid")}
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (4030, 900.0)

    # ---- 1. identify ---------------------------------------------------------
    data_real = load_cmapss_fd001()
    unit = data_real["unit"]
    lo_mask, hi_mask = _era_masks(unit, SPLIT)
    # refit with the explicit split (era_systems uses 0.5 internally; keep one path)
    from sbounds.realsys import SensorSystem
    sys_healthy = SensorSystem.fit(data_real["X"][lo_mask], unit[lo_mask])
    sys_aged = SensorSystem.fit(data_real["X"][hi_mask], unit[hi_mask])
    ident = {
        "n_units_fit": int(unit.max()), "n_cycles": int(data_real["X"].shape[0]),
        "dA_norm": float((sys_healthy.A - sys_aged.A).norm()),
        "dsigma_norm": float((sys_healthy.sigma_vec - sys_aged.sigma_vec).norm()),
        "attractor_shift": float((sys_healthy.equilibrium() - sys_aged.equilibrium()).norm()),
        "clamp_healthy": sys_healthy.clamp_max, "clamp_aged": sys_aged.clamp_max,
        "sigma_healthy": sys_healthy.sigma_vec.tolist(),
        "sigma_aged": sys_aged.sigma_vec.tolist(),
        "sensors": data_real["names"],
    }
    print(f"[ident] ||dA||={ident['dA_norm']:.3f} ||dsigma||={ident['dsigma_norm']:.3f} "
          f"attractor_shift={ident['attractor_shift']:.4f}", flush=True)

    # ---- 2. train on the healthy identified plant ----------------------------
    trained = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else None
    if trained is None:
        system = sys_healthy
        D = system.dim
        # The identified OU model lives in per-cycle time: one data cycle is one
        # unit of time (dt = 1), which is the real sampling of the fleet. This
        # keeps the drift/noise ratio of the training pairs equal to the data's.
        DT = 1.0
        x0 = (2.0 * torch.rand((n_traj, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(0)) - 1.0) \
            * system.lqr_like_scale()
        x1 = euler_maruyama(system, x0, 1, DT,
                            generator=torch.Generator().manual_seed(1))
        data = RolloutData(x0.detach(), x1.detach(), DT)

        transport = InvertibleTransport(
            dim=D, d_latent=D_ETA, n_layers=4, width=32, hidden_depth=2,
            x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=0)
        F = LatentDynamics(D, width=64, depth=2, seed=0, d_eta=D_ETA)
        t0 = time.time()
        wm_hist = train_world_model(transport, F, data, D_ETA,
                                    TrainConfig(steps=wm_steps, seed=0,
                                                w_contract=0.3, d_eta=D_ETA,
                                                rho_spec_cap=1.0))
        t_wm = time.time() - t0

        # region from transported rollout states
        xr = (2.0 * torch.rand((2048, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(2)) - 1.0) \
            * system.lqr_like_scale()
        with torch.no_grad():
            y = transport(xr)
        eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3)
        rho_radius = float(y[:, D_ETA:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
        region = Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                        full_scale=torch.cat([eta_scale,
                                              torch.full((D - D_ETA,),
                                                         rho_radius / (D - D_ETA) ** 0.5,
                                                         dtype=torch.float64)]))

        V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=0)
        t1 = time.time()
        cert_hist = train_certificate(V, F, transport, system, D_ETA,
                                      CertConfig(steps=cert_steps, alpha=ALPHA,
                                                 kappa=KAPPA, seed=0,
                                                 v_res_cap=1.0, v_coef_cap=0.05))
        t_cert = time.time() - t1
        import numpy as _np
        lam_Q = float(_np.linalg.eigvalsh(F.rho_metric(D_ETA).numpy()).max())

        # world-model scores: simulated rollout AND real held-out engines
        xh = (2.0 * torch.rand((256, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(3)) - 1.0) \
            * system.lqr_like_scale()
        # score against a one-cycle push-forward (dt = 1 cycle, the data's unit)
        xh1 = xh + system.drift(xh) * 1.0 \
            + (system.diffusion(xh) @ torch.randn(xh.shape[0], D, 1, dtype=torch.float64,
                                                  generator=torch.Generator().manual_seed(4))).squeeze(-1) * (1.0 ** 0.5)
        from sbounds.systems import pushforward_drift as _pf
        exact = _pf(system, transport, xh)
        with torch.no_grad():
            learned = F(transport(xh))
            rel = float((((learned - exact).norm(dim=-1)) /
                         exact.norm(dim=-1).clamp_min(1e-9)).mean())
        score_sim = {"rel_err_mean": rel}
        Xh, uh = data_real["X_hold"], data_real["unit_hold"]
        keep = uh[1:] == uh[:-1]
        Xr0, Xr1 = Xh[:-1][keep], Xh[1:][keep]
        with torch.no_grad():
            Y0, Y1 = transport(Xr0), transport(Xr1)
            Ypred = Y0 + F(Y0) * 1.0        # F is per unit time; a cycle is the unit
            rel_real = float((((Ypred - Y1).norm(dim=-1)) /
                              (Y1 - Y0).norm(dim=-1).clamp_min(1e-6)).mean())
        score_sim["rel_err_real_holdout"] = rel_real

        from sbounds.generator import noise_floor
        beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)

        trained = {"transport": transport, "F": F, "V": V, "region": region,
                   "system": system,
                   "D": D, "d_eta": D_ETA, "world_model": score_sim,
                   "lam_Q": lam_Q, "beta": beta,
                   "cert_viol_frac": cert_hist["viol_frac"][-1],
                   "cert_max_gen": cert_hist["max_gen"][-1],
                   "seconds_world_model": t_wm, "seconds_certificate": t_cert,
                   "params": sum(p.numel() for p in transport.parameters())
                             + sum(p.numel() for p in F.parameters())
                             + sum(p.numel() for p in V.parameters())}
        torch.save(trained, CKPT)
        print(f"[train] wm={t_wm:.0f}s cert={t_cert:.0f}s "
              f"fit_sim={score_sim.get('rel_err_mean', float('nan')):.4f} "
              f"fit_real_holdout={rel_real:.4f} beta={beta:.2e}", flush=True)

    # ---- 3. certify ----------------------------------------------------------
    if trained.get("cert") is None:
        print(f"[cert] factor d={D_ETA} vs full D={trained['D']} "
              f"({node_budget} nodes each)...", flush=True)
        trained["cert"] = certify_both(trained, node_budget, time_budget,
                                       alpha=ALPHA, tol=TOL)
        torch.save(trained, CKPT)
    c = trained["cert"]
    print(f"[cert] factor {c['factor']['certified_fraction']:.3f} "
          f"({c['factor']['nodes']} nodes, {c['factor']['seconds']:.0f}s) | "
          f"full {c['full']['certified_fraction']:.3f} "
          f"({c['full']['nodes']} nodes, {c['full']['seconds']:.0f}s)", flush=True)

    # ---- 4. era separation ---------------------------------------------------
    if trained.get("era") is None:
        trained["era"] = era_probe(trained, sys_healthy, sys_aged)
        torch.save(trained, CKPT)
    print(f"[era] viol healthy={trained['era']['healthy']['viol_frac']:.3f} "
          f"aged={trained['era']['aged']['viol_frac']:.3f}", flush=True)

    # ---- 5. gated hot-swap with the REAL aged plant as the drift -------------
    if trained.get("shadow") is None:
        region = trained["region"]
        rlo, rhi = region.init_box("full", scale=1.0)
        rlo, rhi = rlo.squeeze(0), rhi.squeeze(0)   # (D,) as exp2 passes them
        shadow = {}
        for gate in ("sound", "naive"):
            t2 = time.time()
            rep = shadow_run(sys_healthy, trained["V"], trained["F"],
                             trained["transport"], KAPPA, ALPHA, D_ETA,
                             rlo, rhi, drift_system=sys_aged,
                             cfg=ShadowConfig(gate=gate, dt=1.0, seed=0))
            shadow[gate] = {"accepted": rep.accepted, "rejected": rep.rejected,
                            "unsafe_swaps": rep.unsafe_swaps,
                            "unsafe_swap_fraction": rep.unsafe_swap_fraction,
                            "fallback_rounds": rep.fallback_rounds,
                            "seconds": time.time() - t2,
                            "rounds": rep.rounds}
            print(f"[shadow:{gate}] accepted={rep.accepted} rejected={rep.rejected} "
                  f"unsafe_swaps={rep.unsafe_swaps} "
                  f"fallback={rep.fallback_rounds} ({time.time() - t2:.0f}s)", flush=True)
        trained["shadow"] = shadow
        torch.save(trained, CKPT)

    # ---- results --------------------------------------------------------------
    out = {"dataset": "NASA C-MAPSS FD001 (Kaggle behrad3d/nasa-cmaps)",
           "identification": ident, "alpha": ALPHA, "kappa": KAPPA, "tol": TOL,
           "d_eta": D_ETA, "D": trained["D"], "beta": trained["beta"],
           "params": trained["params"], "world_model": trained["world_model"],
           "cert_viol_frac": trained["cert_viol_frac"],
           "cert_max_gen": trained["cert_max_gen"],
           "lam_Q": trained["lam_Q"],
           "seconds_world_model": trained["seconds_world_model"],
           "seconds_certificate": trained["seconds_certificate"],
           "cert": trained["cert"], "era": trained["era"],
           "shadow": trained["shadow"]}
    with open(os.path.join(RESULTS, "exp4_realtrend.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp4_realtrend.json", flush=True)


if __name__ == "__main__":
    main()
