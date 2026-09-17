"""exp12 — grand-challenge fluid domain: Karman vortex street (random vortex method).

The fluid-physics entry of the campaign, and the first plant whose dynamics are
a genuine Navier-Stokes scheme: Chorin's random vortex method. A staggered
Karman vortex street is carried by n_v discrete vortices whose positions obey
the 2D Euler Biot-Savart dynamics (exact for point vortices), regularised by
Lamb-Oseen cores, closed by an observation-window flushing term, and driven by
Chorin's Brownian core walk (viscosity as sqrt(2 nu) dW) -- the classical
particle discretisation of 2D Navier-Stokes. D = 2 n_v = 100.

Nominal plant:  n_v = 50, b/h = 0.281 (the classical Karmann optimum), nu = 2e-4.
Drift plant:    b/h = 0.45 with nu = 1.6e-3 -- a street that has moved off its
                stability optimum and into a much more diffusive regime, the
                fluid-dynamical analogue of the actuator/degradation shifts the
                other domains test.

Protocol identical to the other domains: train the invertible transport + latent
world model + Lyapunov certificate on the nominal plant, certify factor-vs-full
at fixed BnB budgets, probe the disturbed street through the same certificate
(era separation), and run the sound shadow gate with the disturbed street as the
real drift. A stable nominal flow with Hurwitz abscissa ~ -0.18 is the honest
operating point for a monitored channel: the certificate should have room to
certify real volume here (unlike the noise-dominated ETTm2 verdict), and the
gate must still reject every swap toward the disturbed street.

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
from sbounds.shadow import (ShadowConfig, shadow_run,
                            true_violation_probe)
from sbounds.systems import pushforward_drift
from sbounds.train import (RolloutData, TrainConfig, CertConfig,
                           euler_maruyama, train_certificate,
                           train_world_model)
from sbounds.transport import InvertibleTransport
from sbounds.vortex import VortexStreet
from experiments.exp1_main import certify_both

CKPT = os.path.join(RESULTS, "exp12_ckpt.pt")
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
D_ETA = 2
REGION_SCALE = 0.5


def make_plants():
    """Nominal street (Karmann optimum) and the disturbed street (drift)."""
    sys_lo = VortexStreet(n_vortices=50, U=1.0, gamma0=1.0, spacing=0.5,
                          core_radius=0.08, nu=2e-4, row_offset=0.281,
                          flush=0.5, seed=0)
    sys_hi = VortexStreet(n_vortices=50, U=1.0, gamma0=1.0, spacing=0.5,
                          core_radius=0.08, nu=1.6e-3, row_offset=0.45,
                          flush=0.5, seed=1)
    return sys_lo, sys_hi


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1500, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (8060, 900.0)

    # ---- 1. plants -----------------------------------------------------------
    sys_lo, sys_hi = make_plants()
    ident = {
        "D": int(sys_lo.dim), "n_vortices": int(sys_lo.n_v),
        "row_offset_nominal": sys_lo.row_offset, "row_offset_drift": sys_hi.row_offset,
        "nu_nominal": sys_lo.nu, "nu_drift": sys_hi.nu,
        "attractor_shift": float((sys_hi.equilibrium() - sys_lo.equilibrium()).norm()),
        "drift_gap": float((sys_hi.drift(sys_lo.equilibrium())
                            - sys_lo.drift(sys_lo.equilibrium())).norm()),
    }
    print(f"[ident] D={ident['D']} drift_gap={ident['drift_gap']:.3f} "
          f"attractor_shift={ident['attractor_shift']:.3f}", flush=True)

    # ---- 2. train on the nominal street --------------------------------------
    trained = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else None
    if trained is None:
        system = sys_lo
        D = system.dim
        DT = 0.05   # vortex-street advection time unit U/h = 1/0.5 -> dt = 0.1 h
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

        # world-model score: simulated push-forward against the exact vortex drift
        xh = (2.0 * torch.rand((256, D), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(3)) - 1.0) \
            * system.lqr_like_scale()
        exact = pushforward_drift(system, transport, xh)
        with torch.no_grad():
            learned = F(transport(xh))
            rel = float((((learned - exact).norm(dim=-1)) /
                         exact.norm(dim=-1).clamp_min(1e-9)).mean())
        score = {"rel_err_mean": rel}

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
              f"fit_sim={rel:.4f} beta={beta:.2e}", flush=True)

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

    # ---- 4. drift separation: nominal street vs the disturbed street ---------
    if trained.get("era") is None:
        region, d = trained["region"], trained["d_eta"]
        lo, hi = region.init_box("full")
        g = torch.Generator().manual_seed(7)
        u = torch.rand((512, trained["transport"].dim), dtype=torch.float64,
                       generator=g)
        y_probe = lo + (hi - lo) * u
        out = {}
        for name, plant in (("nominal", sys_lo), ("disturbed", sys_hi)):
            pr = true_violation_probe(trained["V"], trained["F"],
                                      trained["transport"],
                                      plant, KAPPA, ALPHA, d, y_probe,
                                      beta=trained["beta"], tol=TOL)
            out[name] = {"viol_frac": pr["viol_frac"],
                         "mean_resid": pr.get("mean_resid")}
        trained["era"] = out
        torch.save(trained, CKPT)
    print(f"[era] viol disturbed={trained['era']['disturbed']['viol_frac']:.3f} "
          f"nominal={trained['era']['nominal']['viol_frac']:.3f}", flush=True)

    # ---- 5. gated hot-swap with the disturbed street as the real drift -------
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
                             cfg=ShadowConfig(gate=gate, dt=0.05, seed=0))
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
    out = {"system": "Karman vortex street, Chorin random vortex method "
                     "(n_v=50, Lamb-Oseen cores, window flushing, core-walk nu)",
           "plants": ident, "alpha": ALPHA, "kappa": KAPPA, "tol": TOL,
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
