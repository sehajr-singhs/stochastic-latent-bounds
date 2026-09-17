"""Experiment 9: the learned transport vs fixed linear maps on the NONLINEAR plant.

exp7 established an honest negative-flavored result: on the real-data domains
(the identified plants are linear OU models), ANY orthogonal coordinate map --
PCA, random, or learned -- certifies the same region, because the certificate
machinery only needs *a* factorisation of a linear-Gaussian plant. exp9 closes
the loop on the claim the method actually makes:

    the learned transport matters when the PLANT is nonlinear -- the map must
    undo the sin/cos coupling that a fixed linear projection cannot.

Protocol, identical for all three variants (N-link arm, D=8, d=2, the exp1
headline configuration; same budgets, seeds, region construction, node budget):

- learned: the exp1 arm4 checkpoint (joint invertible transport + latent
  dynamics), re-probed here for pointwise health;
- pca:     fixed orthogonal PCA map from rollout states, F and V trained with
           the exact exp1 recipe through that map;
- random:  fixed random orthogonal map, same protocol.

Results: results/exp9_nonlinear.json
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
from experiments.exp7_baselines import LinearTransport, _pca_U, _random_U
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.shadow import true_violation_probe
from sbounds.systems import ChainArm
from sbounds.train import (CertConfig, TrainConfig, generate_pairs,
                           train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
N_LINKS, D_ETA, SIGMA = 4, 2, 0.05
OUT = os.path.join(RESULTS, "exp9_nonlinear.json")


def _probe_viol(trained: dict, system, n: int = 512, seed: int = 7) -> float:
    region, d = trained["region"], trained["d_eta"]
    lo, hi = region.init_box("full")
    pts = lo + (hi - lo) * torch.rand((n, trained["transport"].dim),
                                      dtype=torch.float64,
                                      generator=torch.Generator().manual_seed(seed))
    pr = true_violation_probe(trained["V"], trained["F"], trained["transport"],
                              system, KAPPA, ALPHA, d, pts,
                              beta=_beta_of(trained), tol=TOL)
    return pr["viol_frac"]


def _beta_of(trained: dict) -> float:
    """Top-level beta for freshly trained variants; cert.beta for exp1 ckpts."""
    b = trained.get("beta")
    if b is None:
        b = trained["cert"]["beta"]
    return float(b)


def run_linear_variant(kind: str, system: ChainArm, wm_steps: int, cert_steps: int,
                       n_traj: int, node_budget: int, time_budget: float,
                       seed: int = 0) -> dict:
    """A fixed linear map through the identical exp1 protocol (dt = 0.01)."""
    t0 = time.time()
    data = generate_pairs(system, n_traj=n_traj, dt=0.01, region_scale=1.0, seed=seed)
    if kind == "pca":
        transport = LinearTransport(_pca_U(system, data), system.equilibrium(),
                                    system.lqr_like_scale())
    else:
        transport = LinearTransport(_random_U(system.dim, 11 + seed), system.equilibrium(),
                                    system.lqr_like_scale())
    F = LatentDynamics(system.dim, width=64, depth=2, seed=seed, d_eta=D_ETA)
    train_world_model(transport, F, data, D_ETA,
                      TrainConfig(steps=wm_steps, seed=seed, w_contract=0.3,
                                  d_eta=D_ETA, rho_spec_cap=1.0))

    xr = generate_pairs(system, n_traj=2048, dt=0.01, seed=123).x0
    with torch.no_grad():
        y = transport(xr)
    eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, D_ETA:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    region = Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([eta_scale,
                                          torch.full((system.dim - D_ETA,),
                                                     rho_radius / (system.dim - D_ETA) ** 0.5,
                                                     dtype=torch.float64)]))

    V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=seed)
    cert_hist = train_certificate(V, F, transport, system, D_ETA,
                                  CertConfig(steps=cert_steps, alpha=ALPHA,
                                             kappa=KAPPA, seed=seed,
                                             v_res_cap=1.0, v_coef_cap=0.05))
    from sbounds.generator import noise_floor
    beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)
    trained = {"transport": transport, "F": F, "V": V, "region": region,
               "system": system, "D": system.dim, "d_eta": D_ETA, "beta": beta}
    cert = certify_both(trained, node_budget, time_budget, alpha=ALPHA, tol=TOL)
    viol = _probe_viol(trained, system, seed=7 + seed)
    return {"cert": cert, "beta": float(beta), "viol_frac": viol,
            "cert_viol_frac": cert_hist["viol_frac"][-1],
            "seconds": time.time() - t0}


def run_learned_variant(system: ChainArm, wm_steps: int, cert_steps: int,
                        n_traj: int, node_budget: int, time_budget: float,
                        seed: int = 0) -> dict:
    """The full learned pipeline (exp1 arm4 protocol) under a seed offset.

    Identical to the committed exp1 protocol: invertible transport (4 blocks,
    width 32) + latent world model with the two-stage refit + quadratic-led
    Lyapunov net + noise-floor beta + certify_both at the standard budget.
    """
    t0 = time.time()
    data = generate_pairs(system, n_traj=n_traj, dt=0.01, region_scale=1.0, seed=seed)
    transport = InvertibleTransport(
        dim=system.dim, d_latent=D_ETA, n_layers=4, width=32, hidden_depth=2,
        x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=seed)
    F = LatentDynamics(system.dim, width=64, depth=2, seed=seed, d_eta=D_ETA)
    train_world_model(transport, F, data, D_ETA,
                      TrainConfig(steps=wm_steps, seed=seed, w_contract=0.3,
                                  d_eta=D_ETA, rho_spec_cap=1.0,
                                  refit_steps=wm_steps // 2))

    xr = generate_pairs(system, n_traj=2048, dt=0.01, seed=123 + seed).x0
    with torch.no_grad():
        y = transport(xr)
    eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, D_ETA:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    region = Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([eta_scale,
                                          torch.full((system.dim - D_ETA,),
                                                     rho_radius / (system.dim - D_ETA) ** 0.5,
                                                     dtype=torch.float64)]))

    V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=seed)
    cert_hist = train_certificate(V, F, transport, system, D_ETA,
                                  CertConfig(steps=3 * cert_steps, alpha=ALPHA,
                                             kappa=KAPPA, seed=seed,
                                             v_res_cap=1.0, v_coef_cap=0.05))
    from sbounds.generator import noise_floor
    beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)
    trained = {"transport": transport, "F": F, "V": V, "region": region,
               "system": system, "D": system.dim, "d_eta": D_ETA, "beta": beta}
    cert = certify_both(trained, node_budget, time_budget, alpha=ALPHA, tol=TOL)
    viol = _probe_viol(trained, system, seed=7 + seed)
    return {"cert": cert, "beta": float(beta), "viol_frac": viol,
            "cert_viol_frac": cert_hist["viol_frac"][-1],
            "seconds": time.time() - t0}


def main() -> None:
    quick = "--quick" in sys.argv
    # --seed N (or SLB_SEED env): offsets every generator by 1000*N so seeds are
    # independent but the default run (seed 0) is bit-identical to the committed one.
    seed_off = 0
    for i, a in enumerate(sys.argv):
        if a == "--seed" and i + 1 < len(sys.argv):
            seed_off = 1000 * int(sys.argv[i + 1])
    seed_off = 1000 * int(os.environ.get("SLB_SEED", seed_off // 1000))
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (400, 400, 800) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1500, 240.0) if quick else (4030, 900.0)

    system = ChainArm(n_links=N_LINKS, sigma=SIGMA)
    store = {}
    if os.path.exists(OUT) and not quick:
        with open(OUT, encoding="utf-8") as fh:
            store = json.load(fh)
    if seed_off and os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            _base = json.load(fh)
        store = {k: v for k, v in _base.items() if not k.startswith("seed")}

    # learned column: committed exp1 arm4 checkpoint (seed 0) or fresh training
    # under the seed offset for the multi-seed runs.
    skey = f"seed{seed_off // 1000}" if seed_off else ""
    ck_dir = os.path.join(RESULTS, "exp1_ckpt")
    ckpt_path = os.path.join(ck_dir, "arm4_d2.pt")
    learned = store.get("learned")
    if seed_off == 0 and os.path.exists(ckpt_path) and (learned is None or quick):
        trained = torch.load(ckpt_path, weights_only=False)
        learned = {"cert": trained["cert"],
                   "beta": float(trained["cert"]["beta"]),
                   "viol_frac": _probe_viol(trained, system),
                   "cert_viol_frac": trained["cert_viol_frac"],
                   "seconds": None}
    elif seed_off:
        lkey = f"seed{seed_off // 1000}_learned"
        if lkey not in store:
            print(f"[learned:{lkey}] full pipeline, seed offset {seed_off}...",
                  flush=True)
            learned = run_learned_variant(system, wm_steps, cert_steps, n_traj,
                                          node_budget, time_budget,
                                          seed=seed_off // 1000)
            store[lkey] = learned
            with open(OUT, "w", encoding="utf-8") as fh:
                json.dump(store, fh, indent=2, default=_jsonable)
            print(f"[learned:{lkey}] factor="
                  f"{learned['cert']['factor']['certified_fraction']:.3f} "
                  f"viol={learned['viol_frac']:.3f}", flush=True)
    if learned is not None and seed_off == 0:
        store["learned"] = learned
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, default=_jsonable)
        print(f"[learned] factor={learned['cert']['factor']['certified_fraction']:.3f} "
              f"viol={learned['viol_frac']:.3f}", flush=True)
    if learned is None and seed_off == 0:
        print("[learned] exp1 arm4 checkpoint not found on this box; "
              "run exp1_main.py first", flush=True)

    for kind in ("pca", "random"):
        key = f"{skey}_{kind}" if skey else kind
        if key in store and not quick:
            continue
        print(f"[{key}] training + certifying ({node_budget} nodes)...", flush=True)
        r = run_linear_variant(kind, system, wm_steps, cert_steps, n_traj,
                               node_budget, time_budget,
                               seed=seed_off // 1000)
        store[key] = r
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, default=_jsonable)
        print(f"[{key}] factor={r['cert']['factor']['certified_fraction']:.3f} "
              f"full={r['cert']['full']['certified_fraction']:.3f} "
              f"viol={r['viol_frac']:.3f} beta={r['beta']:.2e} "
              f"({r['seconds']:.0f}s)", flush=True)

    print("saved results/exp9_nonlinear.json", flush=True)


if __name__ == "__main__":
    main()
