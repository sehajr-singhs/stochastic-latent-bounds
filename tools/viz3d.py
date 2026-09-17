"""The geometry of the certificate — the three-frame visualization.

Frame 1 (wild reality):  the raw physical state of the 4-link arm, projected to
                         the first three joint angles — chaotic, noisy spaghetti.
Frame 2 (the unfolding): the learned invertible transport T acting on a regular
                         grid of physical states — the "hands" that straighten
                         the dynamics. The grid is regular in (q1, q2) with all
                         other coordinates at equilibrium; the image shows the
                         warped (eta_1, eta_2) coordinates.
Frame 3 (the shell):     the same trajectories in latent coordinates
                         (eta_1, eta_2, ||rho||), inside the certified region —
                         the product of the eta-box and the rho-ball — drawn as
                         a transparent shell the trajectories can bounce around
                         in but never puncture. The certified fraction of this
                         region is the committed exp1 number (64.9%).

Also renders an orbiting GIF of frame 3 (the "dotted shape" hero).

Usage:  python tools/viz3d.py [--gif]
Needs:  results/exp1_ckpt/arm4_d2.pt (committed)
Output: docs/assets/triptych.png, docs/assets/shell_orbit.gif
"""

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from sbounds.train import euler_maruyama  # noqa: E402

INK, BLUE, GREEN, GREY, ACCENT = "#1a1a1a", "#267CB9", "#1a7f37", "#8a8a8a", "#c62f2f"


def _load_checkpoint():
    path = os.path.join(ROOT, "results", "exp1_ckpt", "arm4_d2.pt")
    if not os.path.exists(path):
        print(f"checkpoint not found: {path}")
        return None
    return torch.load(path, weights_only=False)


def _rollouts(trained, n_traj=8, steps=600, dt=0.01, seed=7):
    """Roll trajectories storing every step: (steps, n, D)."""
    system = trained["system"]
    region = trained["region"]
    g = torch.Generator().manual_seed(seed)
    lo, hi = region.init_box("full")
    u = torch.rand((n_traj, trained["transport"].dim), dtype=torch.float64,
                   generator=g)
    x = lo + (hi - lo) * u
    xs = [x.clone()]
    for _ in range(steps):
        x = euler_maruyama(system, x, 1, dt, generator=g)
        xs.append(x.clone())
    return torch.stack(xs)


def frame1(ax, xs):
    """Raw physical state: first three joint angles vs time, 3-D spaghetti."""
    q = xs[:, :, :3]  # (steps, n, 3)
    for i in range(q.shape[1]):
        ax.plot(q[:, i, 0], q[:, i, 1], q[:, i, 2], lw=1.1, color=BLUE,
                alpha=0.6)
    ax.scatter(q[0, :, 0], q[0, :, 1], q[0, :, 2], s=10, color=ACCENT, zorder=5)
    ax.set_title("1 — physical space (D = 8):\nraw state, no clean boundary",
                 fontsize=10)
    ax.set_xlabel(r"$q_1$", fontsize=8)
    ax.set_ylabel(r"$q_2$", fontsize=8)
    ax.set_zlabel(r"$q_3$", fontsize=8)
    ax.tick_params(labelsize=6)


def frame2(ax, trained):
    """The transport unfolding a regular physical grid into latent coordinates."""
    transport = trained["transport"]
    system = trained["system"]
    n1 = n2 = 15
    span = 0.9 * system.lqr_like_scale()
    g1 = torch.linspace(-span[0], span[0], n1, dtype=torch.float64)
    g2 = torch.linspace(-span[1], span[1], n2, dtype=torch.float64)
    x = system.equilibrium().repeat(n1 * n2, 1)
    Q1, Q2 = torch.meshgrid(g1, g2, indexing="ij")
    x[:, 0] = Q1.reshape(-1)
    x[:, 1] = Q2.reshape(-1)
    with torch.no_grad():
        y = transport(x)
    eta = y[:, :2].reshape(n1, n2, 2)
    for i in range(n1):
        ax.plot(eta[i, :, 0], eta[i, :, 1], color=BLUE, lw=0.9, alpha=0.75)
    for j in range(n2):
        ax.plot(eta[:, j, 0], eta[:, j, 1], color=BLUE, lw=0.9, alpha=0.75)
    # the certified eta-box (the region's factor part)
    region = trained["region"]
    es = region.eta_scale.tolist()
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-es[0], -es[1]), 2 * es[0], 2 * es[1],
                           fill=True, facecolor=GREEN, alpha=0.12,
                           edgecolor=GREEN, lw=1.4))
    ax.set_title("2 — the transport T:\nregular grid unfolded into (η₁, η₂)",
                 fontsize=10)
    ax.set_xlabel(r"$\eta_1$", fontsize=8)
    ax.set_ylabel(r"$\eta_2$", fontsize=8)
    ax.tick_params(labelsize=6)


def frame3(ax, xs, trained, elev=22, azim=-60):
    """Latent trajectories (eta_1, eta_2, ||rho||) inside the certified shell."""
    transport = trained["transport"]
    region = trained["region"]
    with torch.no_grad():
        y = transport(xs.reshape(-1, xs.shape[-1]))
    d = trained["d_eta"]
    eta = y[:, :d].reshape(xs.shape[0], xs.shape[1], d)
    r = y[:, d:].norm(dim=-1).reshape(xs.shape[0], xs.shape[1])
    es = region.eta_scale.tolist()
    rmax = float(region.rho_radius)

    # shell: the eta-box edges extruded over rho in [0, rmax] (12 corners)
    corners = [(a * es[0], b * es[1]) for a in (-1, 1) for b in (-1, 1)]
    for (a1, b1) in corners:
        ax.plot([a1, a1], [b1, b1], [0, rmax], color=GREEN, lw=0.7, alpha=0.8)
    for z in (0.0, rmax):
        for a in (-1, 1):
            ax.plot([a * es[0], -a * es[0]], [es[1], es[1]], [z, z],
                    color=GREEN, lw=0.7, alpha=0.8)
            ax.plot([es[0], es[0]], [es[1], -es[1]], [z, z],
                    color=GREEN, lw=0.7, alpha=0.8)
            ax.plot([-es[0], es[0]], [-es[1], -es[1]], [z, z],
                    color=GREEN, lw=0.7, alpha=0.8)
            ax.plot([-es[0], -es[0]], [-es[1], es[1]], [z, z],
                    color=GREEN, lw=0.7, alpha=0.8)
    # trajectories: dots coloured by time
    n_show = min(xs.shape[1], 6)
    cmap = plt.get_cmap("viridis")
    for i in range(n_show):
        pts = ax.scatter(eta[:, i, 0], eta[:, i, 1], r[:, i],
                         c=range(xs.shape[0]), cmap=cmap, s=4, alpha=0.55,
                         depthshade=False)
    ax.set_title("3 — latent space (d = 2 + ‖ρ‖):\ndots inside the certified shell",
                 fontsize=10)
    ax.set_xlabel(r"$\eta_1$", fontsize=8)
    ax.set_ylabel(r"$\eta_2$", fontsize=8)
    ax.set_zlabel(r"$\Vert\rho\Vert$", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)
    return pts


def main() -> None:
    trained = _load_checkpoint()
    if trained is None:
        return
    make_gif = "--gif" in sys.argv
    xs = _rollouts(trained)

    fig = plt.figure(figsize=(13.5, 4.6), dpi=140)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    frame1(ax1, xs)
    ax2 = fig.add_subplot(1, 3, 2)
    frame2(ax2, trained)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    frame3(ax3, xs, trained)
    frac = trained["cert"]["factor"]["certified_fraction"]
    fig.suptitle(
        f"The geometry of the certificate — the transport folds the wild state into a "
        f"region of which {frac:.0%} is soundly certified (exp1, arm4, D=8, d=2)",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(ROOT, "docs", "assets", "triptych.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    if make_gif:
        import imageio.v2 as imageio
        frames = []
        for k in range(24):
            azim = -60 + 15 * k
            fig = plt.figure(figsize=(6.2, 5.0), dpi=110)
            ax = fig.add_subplot(111, projection="3d")
            frame3(ax, xs, trained, azim=azim)
            ax.set_title("")
            fig.tight_layout()
            fig.canvas.draw()
            import numpy as np
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(buf.copy())
            plt.close(fig)
        out_gif = os.path.join(ROOT, "docs", "assets", "shell_orbit.gif")
        imageio.mimsave(out_gif, frames, duration=0.18, loop=0)
        print(f"wrote {out_gif}")


if __name__ == "__main__":
    main()
