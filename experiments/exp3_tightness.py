"""Experiment 3: why the tight Ito trace is the load-bearing fix.

For a family of shrinking boxes around the origin this measures the sound upper
bound on sup(L W + alpha W) under two enclosures of the Ito term:

  cs    : 1/2 ||B||_F^2 ||Hess W||_F  (Cauchy-Schwarz; sound, loose)
  tight : 1/2 tr(BB^T Hess W) as an interval trace (sound, and exact in the
          limit of zero box width)

The certificate threshold is beta + tol with beta = L W(0) the noise floor.
The Cauchy-Schwarz slack does not vanish as the box shrinks, so cs certifies
nothing at any radius; the tight bound's excess over beta vanishes linearly,
so it certifies up to an explicit radius. This experiment quantifies exactly
that, and it is the reason the headline comparison (exp1) is meaningful at all.

Usage:  python experiments/exp3_tightness.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, _jsonable
from sbounds.bounds import bound_full
from sbounds.generator import noise_floor
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.systems import ChainArm
from sbounds.transport import InvertibleTransport


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(os.cpu_count() or 4)

    n_links, d_eta = (2, 1) if quick else (4, 2)
    system = ChainArm(n_links=n_links, sigma=0.05)
    D = system.dim
    transport = InvertibleTransport(
        dim=D, d_latent=d_eta, n_layers=4, width=32, hidden_depth=2,
        x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=0)
    V = LyapunovNet(d_eta, width=32, depth=2, n_res=16, seed=0)
    F = LatentDynamics(D, width=64, depth=2, seed=0)
    with torch.no_grad():
        F.A.copy_(-1.5 * torch.eye(D, dtype=torch.float64))
    F.refresh_rho_metric(d_eta)

    kappa, alpha = 1.0, 0.2
    beta = noise_floor(V, F, transport, system, kappa, d_eta)
    tol = 0.05
    widths = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] if not quick else [0.003, 0.03, 0.3]

    rows = []
    for w in widths:
        lo = torch.full((1, D), -w, dtype=torch.float64)
        hi = -lo
        tight = float(bound_full(V, F, transport, system, kappa, alpha, d_eta,
                                 lo, hi, chunk=8).upper[0])
        cs = float(bound_full(V, F, transport, system, kappa, alpha, d_eta,
                              lo, hi, chunk=8, ito_mode="cs").upper[0])
        rows.append({"width": w, "tight_upper": tight, "cs_upper": cs,
                     "tight_excess": tight - beta, "cs_excess": cs - beta,
                     "tight_certifies": tight <= beta + tol,
                     "cs_certifies": cs <= beta + tol})
        print(f"w={w:<6} tight={tight:.4f} (excess {tight - beta:+.4f}) "
              f"cs={cs:.4f} (excess {cs - beta:+.4f})", flush=True)

    out = {"beta": beta, "tol": tol, "kappa": kappa, "alpha": alpha,
           "n_links": n_links, "d_eta": d_eta, "rows": rows}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "exp3_tightness.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp3_tightness.json", flush=True)


if __name__ == "__main__":
    main()
