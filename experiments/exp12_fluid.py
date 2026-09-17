"""exp12 — grand-challenge fluid domain: Rayleigh-Benard convection sensors.

The fluid-physics entry of the campaign: a real Rayleigh-Benard convection
experiment's 1404 temperature/sensor channels (Kaggle eddardd
continuous-stirred-tank-reactor-domain-adaptation; the same array drives the
reactor-safety framing of exp11 -- it is thermal convection sensor data with an
explicit regime change). Stride-8 subsampling gives D = 176; the final quarter
of the timeline is a genuine regime change: the attractor shifts ~0.3 sd per
channel and the variance triples -- the convective-state transition a fluids
monitor cares about, and a *real* drift for the hot-swap gate.

Protocol identical to exp6 (the ETTm2 template): per-era linear-Gaussian
identification, certificate trained on the calm-era plant, factor-vs-full
certification at the same budgets, era separation probe, and the sound shadow
gate with the *disturbed-era plant as the real drift*. Reported either way:
this plant is noise-dominated like ETTm2, so the honest output is the
(region scale, violation) story and the gate's soundness, not a guaranteed
home run.

Usage:  python experiments/exp12_fluid.py [--quick]
Output: results/exp12_fluid.json (+ checkpoint results/exp12_ckpt.pt)
"""

import json
import os
import sys
import time

import numpy as _np
import torch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbounds.generator import noise_floor
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.realsys import SensorSystem, load_cstr
from sbounds.shadow import (ShadowConfig, shadow_run,
                            true_violation_probe)
from sbounds.systems import pushforward_drift
from sbounds.train import (RolloutData, TrainConfig, CertConfig,
                           euler_maruyama, train_certificate,
                           train_world_model)
from sbounds.transport import InvertibleTransport
from experiments.exp1_main import certify_both

CKPT = os.path.join(RESULTS, "exp12_ckpt.pt")
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
D_ETA = 2
REGION_SCALE = 0.5   # noise-dominated plant: the claimed region matches the
                     # stationary noise ball (same reasoning as exp6); the
                     # honest outputs are the measured violation fractions.


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (4030, 900.0)

    # ---- 1. identify ---------------------------------------------------------
    data_real = load_cstr()
    seg = data_real["unit"]
    sys_lo = SensorSystem.fit(data_real["X"][data_real["era_lo"] == 1.0],   # calm nominal
                              seg[data_real["era_lo"] == 1.0])
    sys_hi = SensorSystem.fit(data_real["X"][data_real["era_hi"] == 1.0],   # disturbed
                              seg[data_real["era_hi"] == 1.0])
    ident = {
        "n_rows": int(data_real["X"].shape[0]),
        "D": int(data_real["D"]), "stride": int(data_real["stride"]),
        "dA_norm": float((sys_hi.A - sys_lo.A).norm()),
        "dA_rel": float((sys_hi.A - sys_lo.A).norm()
                        / sys_lo.A.norm().clamp_min(1e-9)),
        "dsigma_norm": float((sys_hi.sigma_vec - sys_lo.sigma_vec).norm()),
        "attractor_shift": float((sys_hi.equilibrium()
                                  - sys_lo.equilibrium()).norm()),
        "clamp_hi": sys_hi.clamp_max, "clamp_lo": sys_lo.clamp_max,
    }
    print(f"[ident] D={ident['D']} ||dA||={ident['dA_norm']:.2f} "
          f"rel={ident['dA_rel']:.2f} attractor_shift="
          f"{ident['attractor_shift']:.4f}", flush=True)

    # ---- 2. train on the calm-era identified plant ---------------------------
    trained = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else None
    if trained is None:
        system = sys_lo
        D = system.dim
        DT = 1.0   # one sample step = one unit of time (the data's real sampling)
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
        train_world_model(transport, F, data, D_ETA,
                          TrainConfig(steps=wm_steps, seed=0, w_contract=0.3,
                                      d_eta=D_ETA, rho_spec_cap=1.0,
                                      refit_steps=wm_steps // 2))
        t_wm = time.time() - t0

        xr = (2.0 * torch.rand((2048, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(2)) - 1.0) \
            * system.lqr_like_scale()
        with torch.no_grad():
            y = transport(xr)
        eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3) \
            * REGION_SCALE
        rho_radius = float((y[:, D_ETA:].norm(dim=-1).quantile(0.9)
                            * REGION_SCALE).clamp_min(1e-3))
        region = Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                        full_scale=torch.cat([
                            eta_scale,
                            torch.full((D - D_ETA,), rho_radius / (D - D_ETA) ** 0.5,
                                       dtype=torch.float64)]))

        V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=0)
        t1 = time.time()
        cert_hist = train_certificate(V, F, transport, system, D_ETA,
                                      CertConfig(steps=3 * cert_steps, alpha=ALPHA,
                                                 kappa=KAPPA, seed=0,
                                                 v_res_cap=1.0, v_coef_cap=0.05))
        t_cert = time.time() - t1
        lam_Q = float(_np.linalg.eigvalsh(F.rho_metric(D_ETA).numpy()).max())

        # world-model scores: simulated push-forward AND real held-out timeline
        xh = (2.0 * torch.rand((256, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(3)) - 1.0) \
            * system.lqr_like_scale()
        exact = pushforward_drift(system, transport, xh)
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

    # ---- 4. era separation: calm plant vs the disturbed regime ---------------
    if trained.get("era") is None:
        region, d = trained["region"], trained["d_eta"]
        lo, hi = region.init_box("full")
        g = torch.Generator().manual_seed(7)
        u = torch.rand((512, trained["transport"].dim), dtype=torch.float64,
                       generator=g)
        y_probe = lo + (hi - lo) * u
        out = {}
        for name, plant in (("calm", sys_lo), ("disturbed", sys_hi)):
            pr = true_violation_probe(trained["V"], trained["F"],
                                      trained["transport"],
                                      plant, KAPPA, ALPHA, d, y_probe,
                                      beta=trained["beta"], tol=TOL)
            out[name] = {"viol_frac": pr["viol_frac"],
                         "mean_resid": pr.get("mean_resid")}
        trained["era"] = out
        torch.save(trained, CKPT)
    print(f"[era] viol disturbed={trained['era']['disturbed']['viol_frac']:.3f} "
          f"calm={trained['era']['calm']['viol_frac']:.3f}", flush=True)

    # ---- 5. gated hot-swap with the REAL disturbed regime as the drift -------
    if trained.get("shadow") is None:
        region = trained["region"]
        rlo, rhi = region.init_box("full", scale=1.0)
        rlo, rhi = rlo.squeeze(0), rhi.squeeze(0)
        shadow = {}
        for gate in ("sound", "naive"):
            t2 = time.time()
            rep = shadow_run(sys_lo, trained["V"], trained["F"],
                             trained["transport"], KAPPA, ALPHA, D_ETA,
                             rlo, rhi, drift_system=sys_hi,
                             cfg=ShadowConfig(gate=gate, dt=1.0, seed=0))
            shadow[gate] = {"accepted": rep.accepted, "rejected": rep.rejected,
                            "unsafe_swaps": rep.unsafe_swaps,
                            "unsafe_swap_fraction": rep.unsafe_swap_fraction,
                            "fallback_rounds": rep.fallback_rounds,
                            "seconds": time.time() - t2,
                            "rounds": rep.rounds}
            print(f"[shadow:{gate}] accepted={rep.accepted} "
                  f"rejected={rep.rejected} unsafe_swaps={rep.unsafe_swaps} "
                  f"fallback={rep.fallback_rounds} ({time.time() - t2:.0f}s)",
                  flush=True)
        trained["shadow"] = shadow
        torch.save(trained, CKPT)

    # ---- results --------------------------------------------------------------
    out = {"dataset": "Rayleigh-Benard convection sensor array "
                      "(Kaggle eddardd/cstr-domain-adaptation, stride 8)",
           "identification": ident, "alpha": ALPHA, "kappa": KAPPA, "tol": TOL,
           "region_scale": REGION_SCALE,
           "d_eta": D_ETA, "D": trained["D"], "beta": trained["beta"],
           "params": trained["params"], "world_model": trained["world_model"],
           "cert_viol_frac": trained["cert_viol_frac"],
           "lam_Q": trained["lam_Q"],
           "cert": trained["cert"], "era": trained["era"],
           "shadow": trained["shadow"],
           "seconds_world_model": trained["seconds_world_model"],
           "seconds_certificate": trained["seconds_certificate"]}
    path = os.path.join(RESULTS, "exp12_fluid.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved results/exp12_fluid.json", flush=True)


if __name__ == "__main__":
    main()
