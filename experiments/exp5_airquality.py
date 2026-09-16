"""Experiment 5: the framework on a second real domain -- Beijing air quality.

Same protocol as exp4 (C-MAPSS), different physical system and different drift
mechanism, to show the pipeline is not fleet-specific:

1. IDENTIFY. Ten hourly channels (PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES,
   DEWP, WSPM) of one PRSA station are z-scored on the *non-heating* era only,
   and an Ornstein-Uhlenbeck model is fit per era. The drift event is the
   documented Beijing heating-season regime shift (Nov 15 - Mar 15): a real,
   seasonal redistribution of the atmospheric dynamics, not a synthetic
   parameter change. 'NA' values are linearly interpolated (count reported).

2. TRAIN. Transport + latent world model on rollouts of the identified
   non-heating plant (dt = 1 hour, the data's sampling).

3. CERTIFY. Factor (d=2 of D=10) vs full branch and bound at equal budget.

4. SEASON SEPARATION. Certificate-violation fraction under the non-heating
   plant vs the heating-season plant, same probes.

5. GATED HOT-SWAP. The exp2 shadow protocol with the REAL heating-season plant
   as the drift event.

Results land in results/exp5_airquality.json.

Usage:  python experiments/exp5_airquality.py [--quick]
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
from experiments.exp4_realtrend import era_probe
from sbounds.realsys import SensorSystem, load_prsa_aq
from sbounds.shadow import ShadowConfig, shadow_run
from sbounds.train import (CertConfig, RolloutData, TrainConfig,
                           euler_maruyama, train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region

CKPT = os.path.join(RESULTS, "exp5_ckpt.pt")
STATION = "Aotizhongxin"
D_ETA = 2
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (4030, 900.0)

    # ---- 1. identify ---------------------------------------------------------
    data_real = load_prsa_aq(station=STATION)
    sys_base = SensorSystem.fit(data_real["X"][data_real["era_lo"] == 1.0])
    sys_heat = SensorSystem.fit(data_real["X"][data_real["era_hi"] == 1.0])
    ident = {
        "station": STATION, "channels": data_real["cols"],
        "n_rows": data_real["n_rows"], "n_filled": data_real["n_filled"],
        "frac_heating": data_real["frac_heating"],
        "dA_norm": float((sys_base.A - sys_heat.A).norm()),
        "dsigma_norm": float((sys_base.sigma_vec - sys_heat.sigma_vec).norm()),
        "attractor_shift": float((sys_base.equilibrium()
                                  - sys_heat.equilibrium()).norm()),
        "clamp_base": sys_base.clamp_max, "clamp_heat": sys_heat.clamp_max,
        "sigma_base": sys_base.sigma_vec.tolist(),
        "sigma_heat": sys_heat.sigma_vec.tolist(),
    }
    print(f"[ident] ||dA||={ident['dA_norm']:.3f} ||dsigma||={ident['dsigma_norm']:.3f} "
          f"attractor_shift={ident['attractor_shift']:.4f}", flush=True)

    # ---- 2. train on the baseline (non-heating) plant ------------------------
    trained = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else None
    if trained is None:
        system = sys_base
        D = system.dim
        DT = 1.0                      # one hour per step: the data's sampling
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
        train_world_model(transport, F, data, D_ETA,
                          TrainConfig(steps=wm_steps, seed=0,
                                      w_contract=0.3, d_eta=D_ETA,
                                      rho_spec_cap=1.0))
        t_wm = time.time() - t0

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

        # world-model score: real held-in-hours are all used for fitting; score
        # the push-forward against the identified plant (the verifier's object)
        from sbounds.systems import pushforward_drift as _pf
        xh = (2.0 * torch.rand((256, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(3)) - 1.0) \
            * system.lqr_like_scale()
        exact = _pf(system, transport, xh)
        with torch.no_grad():
            learned = F(transport(xh))
            rel = float((((learned - exact).norm(dim=-1)) /
                         exact.norm(dim=-1).clamp_min(1e-9)).mean())

        from sbounds.generator import noise_floor
        beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)

        trained = {"transport": transport, "F": F, "V": V, "region": region,
                   "system": system, "sys_heat": sys_heat,
                   "D": D, "d_eta": D_ETA, "world_model": {"rel_err_mean": rel},
                   "lam_Q": lam_Q, "beta": beta,
                   "cert_viol_frac": cert_hist["viol_frac"][-1],
                   "cert_max_gen": cert_hist["max_gen"][-1],
                   "seconds_world_model": t_wm, "seconds_certificate": t_cert,
                   "params": sum(p.numel() for p in transport.parameters())
                             + sum(p.numel() for p in F.parameters())
                             + sum(p.numel() for p in V.parameters())}
        torch.save(trained, CKPT)
        print(f"[train] wm={t_wm:.0f}s cert={t_cert:.0f}s "
              f"fit={rel:.4f} beta={beta:.2e}", flush=True)

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

    # ---- 4. season separation -------------------------------------------------
    if trained.get("era") is None:
        trained["era"] = era_probe(trained, sys_base, sys_heat)
        torch.save(trained, CKPT)
    print(f"[era] viol baseline={trained['era']['healthy']['viol_frac']:.3f} "
          f"heating={trained['era']['aged']['viol_frac']:.3f}", flush=True)

    # ---- 5. gated hot-swap with the REAL heating plant as the drift -----------
    if trained.get("shadow") is None:
        region = trained["region"]
        rlo, rhi = region.init_box("full", scale=1.0)
        rlo, rhi = rlo.squeeze(0), rhi.squeeze(0)
        shadow = {}
        for gate in ("sound", "naive"):
            t2 = time.time()
            rep = shadow_run(sys_base, trained["V"], trained["F"],
                             trained["transport"], KAPPA, ALPHA, D_ETA,
                             rlo, rhi, drift_system=sys_heat,
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
    out = {"dataset": f"Beijing PRSA multi-site air quality, {STATION} "
                      "(Kaggle sid321axn/beijing-multisite-airquality-data-set)",
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
    with open(os.path.join(RESULTS, "exp5_airquality.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp5_airquality.json", flush=True)


if __name__ == "__main__":
    main()
