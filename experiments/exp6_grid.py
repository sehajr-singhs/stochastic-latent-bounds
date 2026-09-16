"""Experiment 6: the framework on real grid physics (ETTm2).

Third real domain, same pipeline as exp4 (NASA C-MAPSS) and exp5 (Beijing air
quality) with zero domain-specific code beyond the loader:

1. IDENTIFY. 7 channels of ETTm2 (6 transformer load levels + oil temperature
   OT), 15-min sampling, z-scored on a train window. An OU model is fit on
   within-(day, regime) first differences, separately for the high-load era
   (hours 08-20) and the low-load era (hours 21-07). The era difference is the
   real, measured regime shift of the grid.

2. TRAIN. Invertible transport + latent world model on rollouts of the
   high-load identified plant (the dominant operating regime).

3. CERTIFY. Factor mode (d=2) vs full mode (D=7) at the same node budget.

4. ERA SEPARATION. Certificate-violation fraction under each real regime.

5. GATED HOT-SWAP. The exp2 shadow protocol with the real low-load plant as
   the drift event (a genuine daily regime change of the grid).

Results land in results/exp6_grid.json.

Usage:  python experiments/exp6_grid.py [--quick]
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
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.realsys import SensorSystem, load_ettm2
from sbounds.shadow import ShadowConfig, shadow_run, true_violation_probe
from sbounds.train import (CertConfig, RolloutData, TrainConfig,
                           euler_maruyama, train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

CKPT = os.path.join(RESULTS, "exp6_ckpt.pt")
D_ETA = 2
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (4030, 900.0)

    # ---- 1. identify ---------------------------------------------------------
    data_real = load_ettm2()
    seg = data_real["unit"]
    sys_hi = SensorSystem.fit(data_real["X"][data_real["era_hi"]], seg[data_real["era_hi"]])
    sys_lo = SensorSystem.fit(data_real["X"][data_real["era_lo"]], seg[data_real["era_lo"]])
    ident = {
        "n_rows": int(data_real["X"].shape[0]),
        "dA_norm": float((sys_hi.A - sys_lo.A).norm()),
        "dsigma_norm": float((sys_hi.sigma_vec - sys_lo.sigma_vec).norm()),
        "attractor_shift": float((sys_hi.equilibrium() - sys_lo.equilibrium()).norm()),
        "clamp_hi": sys_hi.clamp_max, "clamp_lo": sys_lo.clamp_max,
        "sigma_hi": sys_hi.sigma_vec.tolist(),
        "channels": data_real["cols"],
    }
    print(f"[ident] ||dA||={ident['dA_norm']:.3f} "
          f"attractor_shift={ident['attractor_shift']:.4f}", flush=True)

    # ---- 2. train on the high-load identified plant --------------------------
    trained = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else None
    if trained is None:
        system = sys_hi
        D = system.dim
        DT = 1.0   # one 15-min sample = one unit of time (the data's real sampling)
        x0 = (2.0 * torch.rand((n_traj, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(0)) - 1.0) \
            * system.lqr_like_scale()
        x1 = euler_maruyama(system, x0, 1, DT,
                            generator=torch.Generator().manual_seed(1))
        data = RolloutData(x0.detach(), x1.detach(), DT, system=system)

        transport = InvertibleTransport(
            dim=D, d_latent=D_ETA, n_layers=4, width=32, hidden_depth=2,
            x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=0)
        F = LatentDynamics(D, width=64, depth=2, seed=0, d_eta=D_ETA)
        t0 = time.time()
        wm_hist = train_world_model(transport, F, data, D_ETA,
                                    TrainConfig(steps=wm_steps, seed=0,
                                                w_contract=0.3, d_eta=D_ETA,
                                                rho_spec_cap=1.0,
                                                target_pushforward=True))
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

        # world-model scores: simulated push-forward AND real held-out timeline
        from sbounds.systems import pushforward_drift as _pf
        xh = (2.0 * torch.rand((256, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(3)) - 1.0) \
            * system.lqr_like_scale()
        exact = _pf(system, transport, xh)
        with torch.no_grad():
            learned = F(transport(xh))
            rel = float((((learned - exact).norm(dim=-1)) /
                         exact.norm(dim=-1).clamp_min(1e-9)).mean())
        Xh, seg_h = data_real["X_hold"], data_real["unit_hold"]
        keep = seg_h[1:] == seg_h[:-1]
        Xr0, Xr1 = Xh[:-1][keep], Xh[1:][keep]
        with torch.no_grad():
            Y0, Y1 = transport(Xr0), transport(Xr1)
            Ypred = Y0 + F(Y0) * DT
            rel_real = float((((Ypred - Y1).norm(dim=-1)) /
                              (Y1 - Y0).norm(dim=-1).clamp_min(1e-6)).mean())
        score = {"rel_err_mean": rel, "rel_err_real_holdout": rel_real}

        from sbounds.generator import noise_floor
        beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)

        trained = {"transport": transport, "F": F, "V": V, "region": region,
                   "system": system,
                   "D": D, "d_eta": D_ETA, "world_model": score,
                   "lam_Q": lam_Q, "beta": beta,
                   "cert_viol_frac": cert_hist["viol_frac"][-1],
                   "cert_max_gen": cert_hist["max_gen"][-1],
                   "seconds_world_model": t_wm, "seconds_certificate": t_cert,
                   "params": sum(p.numel() for p in transport.parameters())
                             + sum(p.numel() for p in F.parameters())
                             + sum(p.numel() for p in V.parameters())}
        torch.save(trained, CKPT)
        print(f"[train] wm={t_wm:.0f}s cert={t_cert:.0f}s "
              f"fit_sim={rel:.4f} fit_real_holdout={rel_real:.4f} "
              f"beta={beta:.2e}", flush=True)

    # ---- 3. certify ----------------------------------------------------------
    if trained.get("cert") is None:
        print(f"[cert] factor d={D_ETA} vs full D={trained['D']} "
              f"({node_budget} nodes each)...", flush=True)
        trained["cert"] = certify_both(trained, node_budget, time_budget,
                                       alpha=ALPHA, tol=TOL)
        torch.save(trained, CKPT)
    c = trained["cert"]
    print(f"[cert] factor {c['factor']['certified_fraction']:.3f} "
          f"({c['factor']['nodes']} nodes) | full "
          f"{c['full']['certified_fraction']:.3f} ({c['full']['nodes']} nodes)",
          flush=True)

    # ---- 4. era separation ---------------------------------------------------
    if trained.get("era") is None:
        region, d = trained["region"], trained["d_eta"]
        lo, hi = region.init_box("full")
        g = torch.Generator().manual_seed(7)
        u = torch.rand((512, trained["transport"].dim), dtype=torch.float64, generator=g)
        y_probe = lo + (hi - lo) * u
        out = {}
        for name, plant in (("high_load", sys_hi), ("low_load", sys_lo)):
            pr = true_violation_probe(trained["V"], trained["F"], trained["transport"],
                                      plant, KAPPA, ALPHA, d, y_probe,
                                      beta=trained["beta"], tol=TOL)
            out[name] = {"viol_frac": pr["viol_frac"],
                         "mean_resid": pr.get("mean_resid")}
        trained["era"] = out
        torch.save(trained, CKPT)
    print(f"[era] viol high={trained['era']['high_load']['viol_frac']:.3f} "
          f"low={trained['era']['low_load']['viol_frac']:.3f}", flush=True)

    # ---- 5. gated hot-swap with the REAL low-load plant as the drift ---------
    if trained.get("shadow") is None:
        region = trained["region"]
        rlo, rhi = region.init_box("full", scale=1.0)
        rlo, rhi = rlo.squeeze(0), rhi.squeeze(0)
        shadow = {}
        for gate in ("sound", "naive"):
            t2 = time.time()
            rep = shadow_run(sys_hi, trained["V"], trained["F"],
                             trained["transport"], KAPPA, ALPHA, D_ETA,
                             rlo, rhi, drift_system=sys_lo,
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
    out = {"dataset": "ETTm2 electricity transformer (Kaggle alaaelmor/ettsmall)",
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
    with open(os.path.join(RESULTS, "exp6_grid.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp6_grid.json", flush=True)


if __name__ == "__main__":
    main()
