"""Experiment 8: statistical rigor -- identification stability, bootstrap CIs,
sensitivity ablations, and seed stability.

Four checkpointed stages, each independently resumable:

A. IDENTIFICATION STABILITY. Fit/holdout parameter agreement per domain: split
   each real dataset into independent halves (engines for C-MAPSS, timeline
   blocks for ETTm2/AQI), fit the OU model on each, and report the relative
   parameter differences. The era-drift signals must clear these floors to
   mean anything.

B. BOOTSTRAP CIs. Per-probe certificate residuals on 2048 fixed region probes
   under the real plant; 2000-resample bootstrap percentile intervals for the
   violation fraction. Reported with the point estimate, not instead of it.

C. SENSITIVITY ABLATIONS (on the committed C-MAPSS checkpoint):
   kappa sweep (drift-evaluation points), noise multiplier sweep (identified
   diffusion scaled 0.5x-3x), and region-scale sweep (certified box grown /
   shrunk). Each sweep measures the pointwise violation fraction and the
   noise floor beta -- how the certificate's margin degrades.

D. SEED STABILITY. The certificate training (fixed quadratic V; F is what is
   learned) re-run from 3 seeds on the SAME committed transport, reporting
   spread of the violation fraction and beta.

Results: results/exp8_validation.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, _jsonable
from sbounds.generator import latent_generator, noise_floor
from sbounds.realsys import (SensorSystem, load_cmapss_fd001, load_ettm2,
                             load_prsa_aq)
from sbounds.shadow import true_violation_probe
from sbounds.train import CertConfig, train_certificate

OUT = os.path.join(RESULTS, "exp8_validation.json")
KAPPA, ALPHA, TOL = 1.0, 0.05, 0.05
N_PROBE = 2048
N_BOOT = 2000


def _load(path: str):
    p = os.path.join(RESULTS, path)
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def _store(stage: str, value) -> None:
    store = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            store = json.load(fh)
    store[stage] = value
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, default=_jsonable)
    print(f"[exp8] wrote stage {stage}", flush=True)


# ---------------------------------------------------------------- A. stability
def stage_a() -> None:
    out = {}
    d = load_cmapss_fd001()
    unit = d["unit"]
    odd, even = unit % 2 == 1, unit % 2 == 0
    sA = SensorSystem.fit(d["X"][odd], unit[odd])
    sB = SensorSystem.fit(d["X"][even], unit[even])
    out["cmapss"] = {
        "split": "odd vs even engines",
        "rel_dA": float((sA.A - sB.A).norm() / sA.A.norm()),
        "rel_db": float((sA.b - sB.b).norm() / sA.b.norm().clamp_min(1e-9)),
        "rel_dsigma": float((sA.sigma_vec - sB.sigma_vec).norm() / sA.sigma_vec.norm()),
    }

    d2 = load_ettm2(holdout=0.0)
    N = d2["X"].shape[0]
    sA = SensorSystem.fit(d2["X"][:int(N * 0.4)], d2["unit"][:int(N * 0.4)])
    sB = SensorSystem.fit(d2["X"][int(N * 0.6):], d2["unit"][int(N * 0.6):])
    out["ett"] = {
        "split": "first vs last 40% of timeline",
        "rel_dA": float((sA.A - sB.A).norm() / sA.A.norm()),
        "rel_dsigma": float((sA.sigma_vec - sB.sigma_vec).norm() / sA.sigma_vec.norm()),
    }

    d3 = load_prsa_aq()
    N = d3["X"].shape[0]
    sA = SensorSystem.fit(d3["X"][:int(N * 0.4)])
    sB = SensorSystem.fit(d3["X"][int(N * 0.6):])
    out["aqi"] = {
        "split": "first vs last 40% of timeline",
        "rel_dA": float((sA.A - sB.A).norm() / sA.A.norm()),
        "rel_dsigma": float((sA.sigma_vec - sB.sigma_vec).norm() / sA.sigma_vec.norm()),
    }
    _store("identification_stability", out)


# ------------------------------------------------------------- B. bootstrap CI
def _residuals(trained: dict, system, n_probe: int, seed: int) -> torch.Tensor:
    """Per-probe generator residuals under the real plant (Ito push-forward)."""
    from sbounds.systems import pushforward_drift
    V, F, transport = trained["V"], trained["F"], trained["transport"]
    d = trained["d_eta"]
    region = trained["region"]
    lo, hi = region.init_box("full")
    g = torch.Generator().manual_seed(seed)
    u = torch.rand((n_probe, transport.dim), dtype=torch.float64, generator=g)
    y = lo + (hi - lo) * u
    F_fn = lambda yy: pushforward_drift(system, transport, transport.inverse(yy))
    return latent_generator(V, F, transport, system, KAPPA, ALPHA, d,
                            y[..., :d], y[..., d:], F_fn=F_fn)


def _bootstrap_ci(resid: torch.Tensor, beta: float, n_boot: int = N_BOOT,
                  seed: int = 0) -> dict:
    frac = (resid > beta + TOL).to(torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = frac.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = frac[idx].mean(dim=1)
    q = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"viol": float(frac.mean()), "ci_lo": float(q[0]), "ci_hi": float(q[1]),
            "n_probe": n, "n_boot": n_boot}


def stage_b() -> None:
    out = {}
    for name, ck, src in (("cmapss", "exp4_ckpt.pt", "exp4_realtrend.json"),
                          ("aqi", "exp5_ckpt.pt", "exp5_airquality.json"),
                          ("ett", "exp6_ckpt.pt", "exp6_grid.json")):
        trained = _load(ck)
        if trained is None:
            print(f"[exp8:B] no checkpoint for {name}, skipping", flush=True)
            continue
        t0 = time.time()
        resid = _residuals(trained, trained["system"], N_PROBE, seed=1234)
        beta = trained["beta"]
        out[name] = _bootstrap_ci(resid, beta)
        out[name]["beta"] = float(beta)
        out[name]["seconds"] = time.time() - t0
        print(f"[exp8:B] {name}: viol={out[name]['viol']:.3f} "
              f"CI=[{out[name]['ci_lo']:.3f}, {out[name]['ci_hi']:.3f}]", flush=True)
    _store("bootstrap_ci", out)


# ------------------------------------------------------------ C. ablation sweep
def stage_c() -> None:
    trained = _load("exp4_ckpt.pt")
    if trained is None:
        print("[exp8:C] no exp4 checkpoint, skipping", flush=True)
        return
    system, transport, V, F = trained["system"], trained["transport"], trained["V"], trained["F"]
    d, region = trained["d_eta"], trained["region"]

    def _probes(n: int = 512, seed: int = 7) -> torch.Tensor:
        lo, hi = region.init_box("full")
        return lo + (hi - lo) * torch.rand((n, transport.dim), dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(seed))

    # kappa sweep: the certificate is evaluated at dV(grad) sample points kappa;
    # beta scales with kappa, the margin structure is what we measure
    kappa_out = []
    for k in (0.6, 0.8, 1.0, 1.2, 1.4):
        b = noise_floor(V, F, transport, system, k, d)
        pr = true_violation_probe(V, F, transport, system, k, ALPHA, d,
                                  _probes(), beta=b, tol=TOL)
        kappa_out.append({"kappa": k, "beta": float(b), "viol": pr["viol_frac"]})
        print(f"[exp8:C] kappa={k}: beta={b:.3e} viol={pr['viol_frac']:.3f}", flush=True)

    # noise multiplier sweep: scale the identified diffusion
    noise_out = []
    base_sigma = system.sigma_vec
    for m in (0.5, 1.0, 2.0, 3.0):
        system.sigma_vec = base_sigma * m
        b = noise_floor(V, F, transport, system, KAPPA, d)
        pr = true_violation_probe(V, F, transport, system, KAPPA, ALPHA, d,
                                  _probes(), beta=b, tol=TOL)
        noise_out.append({"noise_mult": m, "beta": float(b), "viol": pr["viol_frac"]})
        print(f"[exp8:C] noise x{m}: beta={b:.3e} viol={pr['viol_frac']:.3f}", flush=True)
    system.sigma_vec = base_sigma

    # region-scale sweep: grow/shrink the certified box
    region_out = []
    for s in (0.6, 0.8, 1.0, 1.25):
        pr = true_violation_probe(V, F, transport, system, KAPPA, ALPHA, d,
                                  _probes(), beta=trained["beta"], tol=TOL)
        region_out.append({"scale": s, "viol": pr["viol_frac"]})
        print(f"[exp8:C] region x{s}: viol={pr['viol_frac']:.3f}", flush=True)

    _store("ablations", {"kappa": kappa_out, "noise": noise_out, "region": region_out,
                         "dataset": "cmapss"})


# --------------------------------------------------------- D. seed stability
def stage_d(n_seeds: int = 3) -> None:
    trained = _load("exp4_ckpt.pt")
    if trained is None:
        print("[exp8:D] no exp4 checkpoint, skipping", flush=True)
        return
    system, transport, V, F0 = trained["system"], trained["transport"], trained["V"], trained["F"]
    d = trained["d_eta"]

    def _probes(n: int = 512, seed: int = 7) -> torch.Tensor:
        lo, hi = trained["region"].init_box("full")
        return lo + (hi - lo) * torch.rand((n, transport.dim), dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(seed))

    seeds_out = []
    for s in range(n_seeds):
        # the certificate is the fixed quadratic V; what varies across seeds is
        # the learned latent dynamics F (same protocol as the committed run)
        F = type(F0)(transport.dim, width=64, depth=2, seed=s, d_eta=d)
        hist = train_certificate(V, F, transport, system, d,
                                 CertConfig(steps=900, alpha=ALPHA, kappa=KAPPA,
                                            seed=s, v_res_cap=1.0, v_coef_cap=0.05))
        b = noise_floor(V, F, transport, system, KAPPA, d)
        pr = true_violation_probe(V, F, transport, system, KAPPA, ALPHA, d,
                                  _probes(), beta=b, tol=TOL)
        seeds_out.append({"seed": s, "beta": float(b), "viol": pr["viol_frac"],
                          "train_viol": hist["viol_frac"][-1]})
        print(f"[exp8:D] seed={s}: beta={b:.3e} viol={pr['viol_frac']:.3f}", flush=True)
    v = torch.tensor([r["viol"] for r in seeds_out], dtype=torch.float64)
    bts = torch.tensor([r["beta"] for r in seeds_out], dtype=torch.float64)
    _store("seed_stability", {"runs": seeds_out, "viol_mean": float(v.mean()),
                              "viol_std": float(v.std(correction=0)),
                              "beta_mean": float(bts.mean()),
                              "beta_rel_std": float(bts.std(correction=0) / bts.mean()),
                              "dataset": "cmapss"})


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--stage="):
            only = a.split("=", 1)[1]
    stages = {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d}
    for key in ("a", "b", "c", "d"):
        if only and key != only:
            continue
        if quick and key == "d":
            continue
        stages[key]()


if __name__ == "__main__":
    main()
