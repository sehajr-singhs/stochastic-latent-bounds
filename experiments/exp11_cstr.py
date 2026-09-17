"""exp11 — chemical-reactor scale: CSTR sensor array (D = 176).

Fourth real domain (Kaggle, eddardd/continuous-stirred-tank-reactor-domain-
adaptation): 1404-channel reactor sensor array subsampled by stride 8 to
D = 176, 2860 consecutive samples. The regime change is real: the final
quarter of the timeline shifts the attractor (~0.5 sd/channel) and roughly
triples its variance (rel ||dA|| ~ 8 between era fits) -- runaway-adjacent
behaviour, the kind of drift a reactor certificate must catch.

Protocol (exp6 clone, scaled): OU identification per era on the real data,
learned invertible transport (D=176 -> d=8), interval BnB certificate in
factor and full mode at equal budget, era probes against the *real* disturbed
plant, and the sound shadow gate vs the naive swap under that drift.

Usage:  SLB_THREADS=4 python experiments/exp11_cstr.py [--quick]
Output: results/exp11_cstr.json
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

from experiments.exp1_main import certify_both
from sbounds.generator import noise_floor
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.realsys import SensorSystem, load_cstr
from sbounds.region import Region
from sbounds.shadow import true_violation_probe
from sbounds.train import (CertConfig, RolloutData, TrainConfig,
                           train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
D_ETA = 8


def _jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (torch.dtype,)):
        return str(o)
    raise TypeError(repr(o))


def _era_data(X, era_flag, unit, n_max=4000):
    Xe = X[era_flag == 1.0]
    ue = unit[era_flag == 1.0]
    if Xe.shape[0] > n_max:
        Xe, ue = Xe[-n_max:], ue[-n_max:]
    keep = ue[1:] == ue[:-1]
    return RolloutData(Xe[:-1][keep], Xe[1:][keep], 1.0)


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps = 300 if quick else 1400
    cert_steps = 300 if quick else 1400
    node_budget, time_budget = (400, 120.0) if quick else (8000, 900.0)
    n_probe = 256 if quick else 1024

    t0 = time.time()
    data = load_cstr()
    D = data["D"]
    print(f"[cstr] D={D} N={data['n_rows']} stride={data['stride']}", flush=True)

    sys_calm = SensorSystem.fit(data["X"][data["era_lo"] == 1.0],
                                data["unit"][data["era_lo"] == 1.0])
    sys_dist = SensorSystem.fit(data["X"][data["era_hi"] == 1.0],
                                data["unit"][data["era_hi"] == 1.0])
    rel_dA = float((sys_dist.A - sys_calm.A).norm() / sys_calm.A.norm())
    shift = float((sys_dist.equilibrium() - sys_calm.equilibrium()).abs().mean())
    print(f"[cstr] era fits: rel||dA||={rel_dA:.3f} shift={shift:.3f} sd", flush=True)

    # transport + latent dynamics on the calm era (the deployment regime)
    transport = InvertibleTransport(
        dim=D, d_latent=D_ETA, n_layers=4, width=32, hidden_depth=2,
        x_star=sys_calm.equilibrium(), x_scale=sys_calm.lqr_like_scale(), seed=0)
    F = LatentDynamics(D, width=64, depth=2, seed=0, d_eta=D_ETA)
    calm_pairs = _era_data(data["X"], data["era_lo"], data["unit"],
                           n_max=1200 if quick else 2500)
    train_world_model(transport, F, calm_pairs, D_ETA,
                      TrainConfig(steps=wm_steps, seed=0, w_contract=0.3,
                                  d_eta=D_ETA, rho_spec_cap=1.0))
    print(f"[cstr] world model trained ({time.time() - t0:.0f}s)", flush=True)

    # region from calm-era rollouts under the identified calm plant
    g = torch.Generator().manual_seed(11)
    x0 = sys_calm.lqr_like_scale() * (2 * torch.rand((1024, D), dtype=torch.float64,
                                                     generator=g) - 1)
    with torch.no_grad():
        from sbounds.train import euler_maruyama
        xr = euler_maruyama(sys_calm, x0, 20, 1.0, generator=g)
        y = transport(xr)
    eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, D_ETA:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    region = Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([
                        eta_scale,
                        torch.full((D - D_ETA,),
                                   rho_radius / (D - D_ETA) ** 0.5,
                                   dtype=torch.float64)]))

    V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=0)
    hist = train_certificate(V, F, transport, sys_calm, D_ETA,
                             CertConfig(steps=cert_steps, alpha=ALPHA,
                                        kappa=KAPPA, seed=0,
                                        v_res_cap=0.05, v_coef_cap=0.05))
    beta = noise_floor(V, F, transport, sys_calm, KAPPA, D_ETA)
    trained = {"transport": transport, "F": F, "V": V, "region": region,
               "system": sys_calm, "D": D, "d_eta": D_ETA, "beta": beta}
    print(f"[cstr] certificate trained beta={beta:.3e} "
          f"({time.time() - t0:.0f}s)", flush=True)

    cert = certify_both(trained, node_budget, time_budget, alpha=ALPHA, tol=TOL)
    print(f"[cstr] certified: factor={cert['factor']['certified_fraction']:.3f} "
          f"full={cert['full']['certified_fraction']:.3f}", flush=True)

    # era probes: pointwise certificate under the real disturbed plant
    lo, hi = region.init_box("full")
    g2 = torch.Generator().manual_seed(23)
    pts = lo + (hi - lo) * torch.rand((n_probe, D), dtype=torch.float64, generator=g2)
    probes = {}
    for name, syst in (("calm", sys_calm), ("disturbed", sys_dist)):
        pr = true_violation_probe(V, F, transport, syst, KAPPA, ALPHA, D_ETA,
                                  pts, beta=beta, tol=TOL)
        probes[name] = {"viol_frac": pr["viol_frac"],
                        "max_gen": pr.get("max_gen")}
        print(f"[cstr] era probe {name}: viol={pr['viol_frac']:.3f}", flush=True)

    out = {
        "dataset": {"name": "cstr_sensor_array", "source": "Kaggle "
                    "eddardd/continuous-stirred-tank-reactor-domain-adaptation",
                    "D": D, "stride": data["stride"], "n_rows": data["n_rows"],
                    "n_channels_raw": 1404},
        "identification": {"rel_dA": rel_dA, "attractor_shift_sd": shift,
                           "calm_clamp": sys_calm.clamp_max,
                           "disturbed_clamp": sys_dist.clamp_max},
        "alpha": ALPHA, "kappa": KAPPA, "tol": TOL, "d_eta": D_ETA,
        "beta": float(beta),
        "cert": cert,
        "era_probes": probes,
        "seconds": time.time() - t0,
    }
    path = os.path.join(RESULTS, "exp11_cstr.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp11_cstr.json", flush=True)


if __name__ == "__main__":
    main()
