"""exp10 — computational supremacy at D=200.

The curse-of-dimensionality headline: a 100-link planar arm has D = 200 state
coordinates. The identical protocol as exp1 (same transport, same bound family,
same branch-and-bound) certifies a d = 2 factorisation of it, while the
full-dimensional verifier makes no progress at the same budget. A D-sweep
(20 / 50 / 100 links at d = 2) turns the single point into the scaling law:
cost tracks d, D-independence of the factorised verifier.

Deterministic: fixed seeds everywhere, no resume logic — the whole run is
sized to finish in one stretch on thh (~2 h with 4 threads).

Usage:  SLB_THREADS=4 python experiments/exp10_scaling.py [--quick]
Output: results/exp10_scaling.json
"""

import json
import os
import sys
import time

import torch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbounds.generator import noise_floor
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.systems import ChainArm
from sbounds.train import (CertConfig, TrainConfig, generate_pairs,
                           train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport
from experiments.exp1_main import certify_both


def _jsonable(o):
    if isinstance(o, (torch.Tensor,)):
        return o.tolist()
    if isinstance(o, (torch.dtype,)):
        return str(o)
    raise TypeError(repr(o))


def run_arm(n_links: int, d_eta: int, sigma: float, wm_steps: int,
            cert_steps: int, n_traj: int, node_budget: int,
            time_budget: float, seed: int = 0) -> dict:
    """One (D, d) row of the ladder — the exp1 recipe, verbatim."""
    system = ChainArm(n_links=n_links, sigma=sigma)
    transport = InvertibleTransport(
        dim=system.dim, d_latent=d_eta, n_layers=4, width=32, hidden_depth=2,
        x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=seed)
    F = LatentDynamics(system.dim, width=64, depth=2, seed=seed, d_eta=d_eta)

    data = generate_pairs(system, n_traj=n_traj, dt=0.01, region_scale=1.0,
                          seed=seed)
    train_world_model(transport, F, data, d_eta,
                      TrainConfig(steps=wm_steps, seed=seed, w_contract=0.3,
                                  d_eta=d_eta, rho_spec_cap=1.0))

    xr = generate_pairs(system, n_traj=1024, dt=0.01, seed=seed + 100).x0
    with torch.no_grad():
        y = transport(xr)
    eta_scale = y[:, :d_eta].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, d_eta:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    region = Region(d_eta=d_eta, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([
                        eta_scale,
                        torch.full((system.dim - d_eta,),
                                   rho_radius / (system.dim - d_eta) ** 0.5,
                                   dtype=torch.float64)]))

    V = LyapunovNet(d_eta, use_residual=False, p_scale=1.0, seed=seed)
    hist = train_certificate(V, F, transport, system, d_eta,
                             CertConfig(steps=cert_steps, alpha=0.05, kappa=1.0,
                                        seed=seed, v_res_cap=0.05, v_coef_cap=1.0))
    beta = noise_floor(V, F, transport, system, 1.0, d_eta)
    trained = {"transport": transport, "F": F, "V": V, "region": region,
               "system": system, "D": system.dim, "d_eta": d_eta, "beta": beta}
    cert = certify_both(trained, node_budget, time_budget, alpha=0.05, tol=0.05)
    return {
        "D": system.dim, "d_eta": d_eta, "n_links": n_links,
        "beta": float(beta),
        "factor": cert["factor"], "full": cert["full"],
        "cert_viol_frac": hist["viol_frac"][-1],
        "lam_Q": hist.get("lam_Q", [None])[-1] if hist.get("lam_Q") else None,
        "seconds_world_model": hist.get("seconds_world_model"),
        "seconds_certificate": hist.get("seconds_certificate"),
    }


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))

    # Budgets sized for a ~2 h one-shot run on thh (4 threads).
    wm_steps, cert_steps, n_traj = (80, 80, 200) if quick else (1200, 900, 2000)
    node_budget, time_budget = (200, 30.0) if quick else (8000, 900.0)

    # The headline row + the D-sweep rows (d = 2 everywhere).
    rows = [(20, 2), (50, 2), (100, 2)]
    if quick:
        rows = [(10, 2)]

    out = os.path.join(RESULTS, "exp10_scaling.json")
    store = {"config": {"wm_steps": wm_steps, "cert_steps": cert_steps,
                        "n_traj": n_traj, "node_budget": node_budget,
                        "time_budget": time_budget,
                        "sigma": 0.02, "alpha": 0.05, "kappa": 1.0, "tol": 0.05},
             "rows": {}}

    for n_links, d_eta in rows:
        key = f"arm{n_links}_d{d_eta}"
        t0 = time.time()
        print(f"[{key}] D={2 * n_links}: training "
              f"({wm_steps} wm + {cert_steps} cert steps)...", flush=True)
        r = run_arm(n_links, d_eta, 0.02, wm_steps, cert_steps, n_traj,
                    node_budget, time_budget)
        store["rows"][key] = r
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, default=_jsonable)
        print(f"[{key}] factor={r['factor']['certified_fraction']:.3f} "
              f"full={r['full']['certified_fraction']:.3f} "
              f"beta={r['beta']:.2e} "
              f"factor_nodes={r['factor'].get('nodes')} "
              f"({time.time() - t0:.0f}s)", flush=True)

    print("saved results/exp10_scaling.json", flush=True)


if __name__ == "__main__":
    main()
