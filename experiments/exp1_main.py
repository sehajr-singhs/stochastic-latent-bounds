"""Experiment 1: the headline comparison.

Factor-mode certification searches only the d-dimensional eta box while the
transversal residual rho is handled in closed form; full-mode certification
searches the entire D-dimensional box. Same certificate function W (the FIXED
quadratic 1/2 eta^T P eta over the learned latent factor), same region, same
sound bound family. The claim under test is that factor mode certifies a
comparable fraction of the region at a fraction of the node cost as D grows.

Everything is checkpointed: a run that dies can be re-run and skips finished
stages. Results land in results/exp1.json.

Usage:  python experiments/exp1_main.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, _jsonable
from sbounds.bnb import worst_first_bnb
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region, make_bound_fn
from sbounds.systems import ChainArm
from sbounds.train import (CertConfig, TrainConfig, generate_pairs,
                           score_world_model, train_certificate,
                           train_world_model)
from sbounds.transport import InvertibleTransport

CKPT = os.path.join(RESULTS, "exp1_ckpt")


def save_ckpt(name: str, payload: dict) -> None:
    os.makedirs(CKPT, exist_ok=True)
    torch.save(payload, os.path.join(CKPT, f"{name}.pt"))


def load_ckpt(name: str) -> dict | None:
    p = os.path.join(CKPT, f"{name}.pt")
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def build_and_train(n_links: int, d_eta: int, sigma: float, alpha: float,
                    kappa: float, seed: int, wm_steps: int, cert_steps: int,
                    n_traj: int, verbose: bool = False) -> dict:
    """Build the full pipeline for one arm size and train it. Returns models+meta."""
    system = ChainArm(n_links=n_links, sigma=sigma)
    D = system.dim
    data = generate_pairs(system, n_traj=n_traj, dt=0.01, region_scale=1.0, seed=seed)
    transport = InvertibleTransport(
        dim=D, d_latent=d_eta, n_layers=4, width=32, hidden_depth=2,
        x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=seed)
    F = LatentDynamics(D, width=64, depth=2, seed=seed, d_eta=d_eta)
    t0 = time.time()
    # w_contract pushes the learned rho block to be Hurwitz at the stated rate,
    # which the rho metric needs: a slow/non-Hurwitz A_rr makes Q ill-conditioned
    # and the transversal term of the certificate large.
    wm_hist = train_world_model(transport, F, data, d_eta,
                                TrainConfig(steps=wm_steps, seed=seed,
                                            w_contract=0.3, d_eta=d_eta,
                                            rho_spec_cap=1.0), verbose=verbose)
    t_wm = time.time() - t0

    xh = generate_pairs(system, n_traj=256, dt=0.01, seed=seed + 99).x0
    score = score_world_model(system, transport, F, xh)

    # region: 90th percentile of transported coordinates of visited states
    xr = generate_pairs(system, n_traj=2048, dt=0.01, seed=seed + 123).x0
    with torch.no_grad():
        y = transport(xr)
    eta_scale = y[:, :d_eta].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, d_eta:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    region = Region(d_eta=d_eta, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([eta_scale,
                                          torch.full((D - d_eta,), rho_radius / (D - d_eta) ** 0.5,
                                                     dtype=torch.float64)]))

    # The certificate is the FIXED quadratic V = 1/2 eta^T P eta (p_scale = 1):
    # see LyapunovNet's docstring for why this mode is the sound default.
    V = LyapunovNet(d_eta, use_residual=False, p_scale=1.0, seed=seed)
    t1 = time.time()
    cert_hist = train_certificate(V, F, transport, system, d_eta,
                                  CertConfig(steps=cert_steps, alpha=alpha, kappa=kappa,
                                             seed=seed, v_res_cap=1.0, v_coef_cap=0.05),
                                  verbose=verbose)
    t_cert = time.time() - t1
    metric = F.refresh_rho_metric(d_eta)
    import numpy as _np
    lam_Q = float(_np.linalg.eigvalsh(F.rho_metric(d_eta).numpy()).max())
    return {
        "system": system, "transport": transport, "F": F, "V": V, "region": region,
        "D": D, "d_eta": d_eta,
        "world_model": score,
        "final_lam_rho": wm_hist["final_lam_rho"],
        "lam_Q": lam_Q,
        "rho_metric_ok": metric["rho_metric_ok"],
        "cert_viol_frac": cert_hist["viol_frac"][-1],
        "cert_max_gen": cert_hist["max_gen"][-1],
        "seconds_world_model": t_wm, "seconds_certificate": t_cert,
        "params": sum(p.numel() for p in transport.parameters())
                  + sum(p.numel() for p in F.parameters())
                  + sum(p.numel() for p in V.parameters()),
    }


def certify_both(trained: dict, node_budget: int, time_budget: float,
                 chunk: int = 96) -> dict:
    """Certify the same region in factor mode (d coords) and full mode (D coords).

    The threshold is the noise floor beta = L W(0) plus the explicit slack tol:
    with additive process noise sup(LW + aW) exceeds beta on every neighbourhood
    of the origin, so certifying against bare beta would return 0 by
    construction rather than by search failure. The guaranteed invariant set is
    the noise ball of stationary radius (beta + tol) / alpha.
    """
    from sbounds.generator import noise_floor

    V, F = trained["V"], trained["F"]
    transport, system = trained["transport"], trained["system"]
    region, d, alpha, kappa = trained["region"], trained["d_eta"], ALPHA, KAPPA
    beta = noise_floor(V, F, transport, system, kappa, d)
    tol = TOL
    out = {"beta": beta, "tol": tol, "threshold": beta + tol}
    for mode in ("factor", "full"):
        lo, hi = region.init_box(mode)
        bf = make_bound_fn(mode, V, F, transport, system, kappa, alpha, region,
                           chunk=chunk, rho_rings=RHO_RINGS)
        res = worst_first_bnb(bf, lo, hi, region.split_dims(mode), node_budget=node_budget,
                              time_budget=time_budget, return_unknown=False,
                              threshold=beta + tol)
        out[mode] = {"certified_fraction": res.certified_fraction, "nodes": res.nodes,
                     "seconds": res.seconds, "worst_upper": res.worst_upper,
                     "fully_certified": res.fully_certified}
    return out


ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
RHO_RINGS = 8          # shell partition of the rho ball in factor mode


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(os.cpu_count() or 4)
    if quick:
        sizes = [(4, 2)]
        wm_steps, cert_steps, n_traj = 400, 400, 800
        node_budget, time_budget = 1500, 240.0
    else:
        sizes = [(4, 2), (6, 3), (8, 4), (10, 4)]
        wm_steps, cert_steps, n_traj = 1500, 1200, 4000
        node_budget, time_budget = 30000, 2400.0

    rows = []
    for n_links, d_eta in sizes:
        tag = f"arm{n_links}_d{d_eta}"
        trained = load_ckpt(tag)
        if trained is None:
            print(f"[{tag}] training...", flush=True)
            trained = build_and_train(n_links, d_eta, sigma=0.05, alpha=ALPHA,
                                      kappa=KAPPA, seed=0, wm_steps=wm_steps,
                                      cert_steps=cert_steps, n_traj=n_traj)
            trained["cert"] = None
            save_ckpt(tag, trained)
            print(f"[{tag}] wm={trained['seconds_world_model']:.0f}s "
                  f"cert={trained['seconds_certificate']:.0f}s "
                  f"fit={trained['world_model'].get('rel_err_mean', float('nan')):.4f}",
                  flush=True)
        if trained.get("cert") is None:
            print(f"[{tag}] certifying (factor d={d_eta} vs full D={trained['D']})...",
                  flush=True)
            trained["cert"] = certify_both(trained, node_budget, time_budget)
            save_ckpt(tag, trained)
        c = trained["cert"]
        row = {"n_links": n_links, "D": trained["D"], "d_eta": d_eta,
               "params": trained["params"],
               "world_model": trained["world_model"],
               "cert_viol_frac": trained["cert_viol_frac"],
               "final_lam_rho": trained["final_lam_rho"],
               "lam_Q": trained["lam_Q"], "rho_metric_ok": trained["rho_metric_ok"],
               "seconds_world_model": trained["seconds_world_model"],
               "seconds_certificate": trained["seconds_certificate"],
               "beta": c["beta"], "tol": c["tol"], "threshold": c["threshold"],
               "factor": c["factor"], "full": c["full"]}
        rows.append(row)
        print(f"[{tag}] factor frac={c['factor']['certified_fraction']:.3f} "
              f"({c['factor']['nodes']} nodes, {c['factor']['seconds']:.0f}s) | "
              f"full frac={c['full']['certified_fraction']:.3f} "
              f"({c['full']['nodes']} nodes, {c['full']['seconds']:.0f}s)", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = {"alpha": ALPHA, "kappa": KAPPA, "sigma": 0.05, "rows": rows}
    with open(os.path.join(RESULTS, "exp1.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp1.json", flush=True)


if __name__ == "__main__":
    main()
