"""Figure 12: the Karman vortex street domain, rendered from the actual plant.

Three panels, all from direct simulation of sbounds.vortex.VortexStreet
(no artistic license -- every dot is a vortex position the SDE produced):

  A. the steady street: equilibrium positions, colored by circulation sign,
     with the Lamb-Oseen core scale shown as dot size;
  B. the stochastic street: ~40 snapshots of one noisy realization overlaid
     (the "dotted shape" -- the stationary cloud the certificate must bound);
  C. drift magnitude |dx/dt| per vortex for nominal vs disturbed streets:
     the plant drift the shadow gate must detect (b/h 0.281 -> 0.45, nu x8).

Usage:  python tools/vortex_fig.py
Output: docs/assets/fig12_vortex_street.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbounds.vortex import VortexStreet

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "assets", "fig12_vortex_street.png")


def _pos(x: torch.Tensor, n: int) -> np.ndarray:
    return x.unflatten(-1, (n, 2)).cpu().numpy()


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    nominal = VortexStreet(n_vortices=50, row_offset=0.281, nu=2e-4, seed=0)
    disturbed = VortexStreet(n_vortices=50, row_offset=0.45, nu=1.6e-3, seed=1)
    n, DT = nominal.n_v, 0.05

    fig = plt.figure(figsize=(15.0, 4.6), dpi=140)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.28)

    # ---- A: the steady street ------------------------------------------------
    axA = fig.add_subplot(gs[0])
    X = _pos(nominal.X_star.reshape(-1), n)
    sign = nominal._gamma.numpy()
    sizes = 14.0 * (1.0 - np.exp(-(0.05 / nominal.a) ** 2)) + 8.0
    axA.scatter(X[:, 0], X[:, 1], c=np.where(sign > 0, "#d1495b", "#2e86ab"),
                s=sizes, zorder=3, edgecolors="k", linewidths=0.3)
    axA.set_title("A · steady Kármán street (n=50 vortices)", fontsize=10)
    axA.set_xlabel("x / h"); axA.set_ylabel("y / h")
    axA.set_aspect("equal")

    # ---- B: the stochastic street (dotted cloud) ------------------------------
    axB = fig.add_subplot(gs[1])
    x = nominal.equilibrium() + 0.02 * torch.randn(nominal.dim, dtype=torch.float64)
    eye = torch.eye(nominal.dim, dtype=torch.float64)
    sig = (2.0 * nominal.nu) ** 0.5
    snaps = []
    for t in range(1600):
        d = nominal.drift(x)
        x = x + DT * d + sig * (DT ** 0.5) * torch.randn(nominal.dim,
                                                         dtype=torch.float64)
        if t % 40 == 0:
            snaps.append(_pos(x, n))
    cloud = np.concatenate(snaps, axis=0)
    axB.scatter(cloud[:, 0], cloud[:, 1], s=1.2, c="#4a4e69", alpha=0.25,
                zorder=2, linewidths=0)
    Xs = _pos(nominal.X_star.reshape(-1), n)
    axB.scatter(Xs[:, 0], Xs[:, 1], c=np.where(sign > 0, "#d1495b", "#2e86ab"),
                s=10, zorder=3, edgecolors="k", linewidths=0.3)
    axB.set_title(f"B · stochastic street: {len(snaps)} snapshots overlaid "
                  r"($\sqrt{2\nu}$ core walk)", fontsize=10)
    axB.set_xlabel("x / h"); axB.set_ylabel("y / h")
    axB.set_aspect("equal")

    # ---- C: drift magnitude, nominal vs disturbed -----------------------------
    axC = fig.add_subplot(gs[2])
    x0 = nominal.equilibrium() + 0.01 * torch.randn(256, nominal.dim,
                                                    dtype=torch.float64)
    dn = nominal.drift(x0).norm(dim=-1).cpu().numpy()
    dd = disturbed.drift(x0).norm(dim=-1).cpu().numpy()
    for arr, col, lab in ((dn, "#2e86ab", "nominal (b/h=0.281, ν=2e-4)"),
                          (dd, "#d1495b", "disturbed (b/h=0.45, ν=1.6e-3)")):
        hist, edges = np.histogram(arr, bins=40, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        axC.plot(centers, hist, color=col, lw=1.8, label=lab)
    axC.set_title("C · drift-speed distributions (the gate's signal)", fontsize=10)
    axC.set_xlabel(r"$|\,\mathrm{dx}/\mathrm{dt}\,|$ per vortex")
    axC.set_ylabel("density")
    axC.legend(fontsize=8, frameon=False)

    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=8)
        for s in ax.spines.values():
            s.set_linewidth(0.5)

    fig.suptitle("exp12 plant: 2D Navier–Stokes by Chorin's random vortex method "
                 "(Biot–Savart drift, Lamb–Oseen cores, window flushing)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
