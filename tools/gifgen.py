"""Animated figures for the site, in the e3nn.org style.

Two GIFs, both rendered from the committed trained checkpoint:

1. hero_latent.gif  -- the site's hero animation: the certified factor region
   in latent space. Blue = certified sub-boxes from the real branch-and-bound
   run, warm = boxes the sound bound could not yet separate, and a slowly
   evolving cloud of latent trajectories of the identified plant. This is the
   picture the method sells: a region that provably shrinks.

2. bnb_subdivide.gif -- the algorithm's loop, animated: worst-first branch and
   bound subdividing the latent region, with the certified fraction climbing
   and the worst upper bound falling as boxes resolve.

Everything is deterministic (fixed seeds) and re-runnable:
    python tools/gifgen.py [--quick]
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS
from sbounds.bnb import worst_first_bnb
from sbounds.region import make_bound_fn

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
ASSETS = os.path.join(DOCS, "assets")
CKPT = os.path.join(RESULTS, "cascade_arm4.pt")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _load_trained():
    ckpt = os.path.join(RESULTS, "exp1_ckpt", "arm4.pt")
    if not os.path.exists(ckpt):
        ckpt = CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError("no arm4 checkpoint found")
    t = torch.load(ckpt, weights_only=False, map_location="cpu")
    if t.get("beta") is None:
        from sbounds.generator import noise_floor
        t["beta"] = noise_floor(t["V"], t["F"], t["transport"], t["system"],
                                1.0, t["d_eta"])
    return t


def _palette():
    blue = "#2b6cb0"
    warm = "#dd6b20"
    ink = "#1a202c"
    return blue, warm, ink


def _collect_boxes(trained, frames: int, node_budget: int):
    """Run worst-first BnB, snapshotting the frontier at `frames` points."""
    region = trained["region"]
    lo, hi = region.init_box("factor")
    lo, hi = lo.squeeze(0), hi.squeeze(0)

    from sbounds.bnb import BnBResult  # noqa: F401  (contract reference)

    snapshots = []
    # Re-run BnB in slices: after each slice, record (boxes certified so far,
    # worst un-resolved boxes). worst_first_bnb returns unresolved boxes, so we
    # drive it manually via the bound function to capture the full state.
    # manual replay of the heap loop to grab snapshots
    import heapq
    bound_fn = make_bound_fn("factor", trained["V"], trained["F"],
                             trained["transport"], trained["system"],
                             1.0, 0.05, region=region)
    t0 = __import__("time").time()
    up0 = bound_fn(lo.unsqueeze(0), hi.unsqueeze(0)).reshape(-1)
    threshold = float(trained["beta"]) + 0.05
    heap = [(-float(up0.max()), 0, 0, lo, hi, up0)]
    uid = 1
    certified = []           # list of (lo, hi, upper) certified boxes
    pending = heap           # frontier
    nodes = 0
    per_frame = node_budget // frames
    for f in range(frames):
        target = (f + 1) * per_frame
        while pending and nodes < target:
            neg, tie, _, blo, bhi, bup = heapq.heappop(pending)
            nodes += 1
            w = int(torch.argmax((bhi - blo)).item())
            mid = 0.5 * (blo[w] + bhi[w])
            loA, hiA = blo.clone(), bhi.clone(); hiA[w] = mid
            loB, hiB = blo.clone(), bhi.clone(); loB[w] = mid
            ups = bound_fn(torch.stack([loA, loB]), torch.stack([hiA, hiB])).reshape(-1)
            for (cl, ch, up) in ((loA, hiA, ups[0]), (loB, hiB, ups[1])):
                if float(up) <= threshold:
                    certified.append((cl, ch, float(up)))
                else:
                    heapq.heappush(pending, (-float(up), uid, uid, cl, ch, up))
                    uid += 1
        snapshots.append({"certified": [(a.clone(), b.clone(), u) for a, b, u in certified],
                          "pending": [(c[3].clone(), c[4].clone()) for c in pending[:200]],
                          "nodes": nodes,
                          "worst": -pending[0][0] if pending else float("nan"),
                          "seconds": __import__("time").time() - t0})
    return snapshots


def hero_gif(trained, path: str, frames: int = 40, node_budget: int = 3000) -> None:
    blue, warm, ink = _palette()
    region = trained["region"]
    d = trained["d_eta"]
    system = trained["system"]
    transport = trained["transport"]
    V, F = trained["V"], trained["F"]

    snapshots = _collect_boxes(trained, frames=frames, node_budget=node_budget)

    # latent trajectories of the identified plant for the ambient cloud
    g = torch.Generator().manual_seed(3)
    eta_lo, eta_hi = region.init_box("factor")
    eta_lo, eta_hi = eta_lo.squeeze(0), eta_hi.squeeze(0)
    D = trained["D"]
    x0 = (2 * torch.rand((64, D), dtype=torch.float64, generator=g) - 1) \
        * system.lqr_like_scale()
    with torch.no_grad():
        y0 = transport(x0)
    trajs = [y0[:, :2].numpy()]
    y = y0
    dt = 0.4
    with torch.no_grad():
        for _ in range(24):
            dy = F(y) * dt
            y = y + torch.cat([dy[:, :2],
                               torch.zeros_like(y[:, 2:])], dim=-1)
            trajs.append(y[:, :2].numpy())
    trajs = np.stack(trajs)            # (T, N, 2)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.set_xlim(float(eta_lo[0]) * 1.1, float(eta_hi[0]) * 1.1)
    ax.set_ylim(float(eta_lo[1]) * 1.1, float(eta_hi[1]) * 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ink); s.set_linewidth(0.8)
    ax.set_title("certified latent region (d = 2)", color=ink, fontsize=11)

    def draw(k):
        ax.cla()
        ax.set_xlim(float(eta_lo[0]) * 1.1, float(eta_hi[0]) * 1.1)
        ax.set_ylim(float(eta_lo[1]) * 1.1, float(eta_hi[1]) * 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(ink); sp.set_linewidth(0.8)
        ax.set_title("certified latent region (d = 2)", color=ink, fontsize=11)
        snap = snapshots[min(k, len(snapshots) - 1)]
        for (blo, bhi, up) in snap["certified"]:
            ax.add_patch(Rectangle((float(blo[0]), float(blo[1])),
                                   float(bhi[0] - blo[0]), float(bhi[1] - blo[1]),
                                   facecolor=blue, alpha=0.25, edgecolor=blue,
                                   linewidth=0.4))
        for (blo, bhi) in snap["pending"][:150]:
            ax.add_patch(Rectangle((float(blo[0]), float(blo[1])),
                                   float(bhi[0] - blo[0]), float(bhi[1] - blo[1]),
                                   facecolor="none", edgecolor=warm,
                                   linewidth=0.7, alpha=0.8))
        T = trajs.shape[0]
        start = (k * 1) % T
        seg = np.concatenate([trajs[(start + i) % T] for i in range(8)], axis=0)
        ax.scatter(seg[:, 0], seg[:, 1], s=3, color=ink, alpha=0.28, linewidths=0)
        n_cert = len(snap["certified"])
        n_pend = len(snap["pending"])
        frac = n_cert / max(n_cert + n_pend, 1)
        ax.text(0.02, 0.02, f"nodes {snap['nodes']}  certified {frac:.0%}",
                transform=ax.transAxes, fontsize=9, color=ink)

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=120)
    anim.save(path, writer=animation.PillowWriter(fps=9))
    plt.close(fig)
    print(f"wrote {os.path.basename(path)}  ({frames} frames, {node_budget} nodes)")


def bnb_gif(trained, path: str, frames: int = 24, node_budget: int = 2000) -> None:
    """The subdivision loop, abstracted: certified volume vs unresolved frontier."""
    blue, warm, ink = _palette()
    snapshots = _collect_boxes(trained, frames=frames, node_budget=node_budget)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=110)

    def draw(k):
        for ax in (ax1, ax2):
            ax.cla()
        snap = snapshots[min(k, len(snapshots) - 1)]
        region = trained["region"]
        lo, hi = region.init_box("factor")
        lo, hi = lo.squeeze(0), hi.squeeze(0)
        vol_cert = sum(float((b - a).prod()) for a, b, _ in snap["certified"])
        vol_total = float((hi - lo).prod())
        frac = vol_cert / vol_total
        # left: bar of certified volume
        ax1.bar([0], [frac * 100], color=blue, width=0.55)
        ax1.bar([0], [(1 - frac) * 100], bottom=[frac * 100], color=warm,
                alpha=0.35, width=0.55)
        ax1.set_ylim(0, 100)
        ax1.set_xticks([])
        ax1.set_ylabel("certified volume (%)", color=ink)
        ax1.set_title(f"branch & bound, {snap['nodes']} nodes", color=ink, fontsize=10)
        # right: worst sound upper bound vs threshold
        thr = float(trained["beta"]) + 0.05
        ax2.axhline(thr, color="green", linestyle="--", linewidth=1.2,
                    label="threshold $\\beta+\\tau$")
        hist = [s["worst"] for s in snapshots[:k + 1]]
        ax2.plot(range(len(hist)), hist, color=ink, linewidth=1.6)
        ax2.set_xlabel("BnB round", color=ink)
        ax2.set_ylabel("worst sound upper bound", color=ink)
        ax2.legend(frameon=False, fontsize=8)

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=160)
    anim.save(path, writer=animation.PillowWriter(fps=7))
    plt.close(fig)
    print(f"wrote {os.path.basename(path)}")


def main() -> None:
    quick = "--quick" in sys.argv
    os.makedirs(ASSETS, exist_ok=True)
    trained = _load_trained()
    hero_gif(trained, os.path.join(ASSETS, "hero_latent.gif"),
             frames=8 if quick else 40,
             node_budget=600 if quick else 3000)
    bnb_gif(trained, os.path.join(ASSETS, "bnb_subdivide.gif"),
            frames=6 if quick else 24,
            node_budget=400 if quick else 2000)


if __name__ == "__main__":
    main()
