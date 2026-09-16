"""Experiment 7: transport baselines -- does the *learned* map matter?

The claim under test: the certificate works because the transport is *learned*
(an invertible diffeomorphism trained jointly with the latent dynamics), not
because any low-dimensional projection would do. Three coordinate maps, one
downstream protocol:

- pca:    fixed orthogonal map whose first d axes are the top-d principal
          components of rollout states ("just use PCA")
- random: fixed random orthogonal map ("any subspace" strawman)
- learned: the method's InvertibleTransport -- numbers imported from the
          committed exp4/exp5/exp6 results (identical seeds, budgets, and
          protocol; re-training here would only duplicate compute)

The fixed maps implement the exact transport interface the certificate
machinery consumes (forward/inverse/r_zero/interval_forward/interval_inverse/
interval_jacobian/interval_jacobian_frobenius/logdet), with *exact* linear
interval arithmetic -- so the sound bounds remain sound and the only changed
variable is the coordinate map. Everything downstream (latent-dynamics
training budget, fixed-quadratic certificate, region construction, node
budget, probe seeds) is identical to the learned runs.

Results: results/exp7_baselines.json
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
from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.realsys import (SensorSystem, load_cmapss_fd001, load_ettm2,
                             load_prsa_aq)
from sbounds.shadow import true_violation_probe
from sbounds.train import (CertConfig, RolloutData, TrainConfig,
                           euler_maruyama, train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

CKPT = os.path.join(RESULTS, "exp7_ckpt.pt")
D_ETA = 2
ALPHA, KAPPA, TOL = 0.05, 1.0, 0.05
RHO_RINGS = 8
LEARNED_SRC = {"cmapss": "exp4_realtrend.json", "aqi": "exp5_airquality.json",
               "ett": "exp6_grid.json"}


class LinearTransport(torch.nn.Module):
    """Fixed linear coordinate map y = U^T (x - x*) / x_scale, U orthogonal.

    The certified factor is y[..., :d] -- the same split convention as the
    learned transport. Interval arithmetic is exact for linear maps.
    """

    noise_freeze = True

    def __init__(self, U: torch.Tensor, x_star: torch.Tensor, x_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("U", U.to(torch.float64))            # (D, D)
        self.register_buffer("x_star", x_star.to(torch.float64))
        self.register_buffer("x_scale", x_scale.to(torch.float64))
        self.dim = U.shape[0]
        # Adam requires a parameter; frozen so the map never moves
        self._anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64),
                                          requires_grad=False)

    def _norm(self, x):
        return (x - self.x_star) / self.x_scale

    def r_zero(self) -> torch.Tensor:
        return torch.zeros(self.dim, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) @ self.U

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (y @ self.U.T) * self.x_scale + self.x_star

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return -self.x_scale.log().sum().expand(x.shape[:-1])

    # --- exact linear interval maps ---------------------------------------
    @torch.no_grad()
    def interval_forward(self, x_iv):  # noqa: ANN001 - Iv type lives in nets
        from sbounds.nets import Iv
        xn_lo, xn_hi = self._norm(x_iv.lo), self._norm(x_iv.hi)
        a, b = xn_lo @ self.U, xn_hi @ self.U
        return Iv(torch.minimum(a, b), torch.maximum(a, b))

    @torch.no_grad()
    def interval_inverse(self, y_iv):
        from sbounds.nets import Iv
        a, b = y_iv.lo @ self.U.T, y_iv.hi @ self.U.T
        lo = torch.minimum(a, b) * self.x_scale + self.x_star
        hi = torch.maximum(a, b) * self.x_scale + self.x_star
        return Iv(lo, hi)

    @torch.no_grad()
    def interval_jacobian(self, x_iv):
        from sbounds.nets import Iv
        B = x_iv.lo.shape[0]
        J = (self.U.T / self.x_scale).expand(B, self.dim, self.dim).contiguous()
        return Iv(J.clone(), J.clone())

    @torch.no_grad()
    def interval_jacobian_frobenius(self, x_iv) -> torch.Tensor:
        J = self.U.T / self.x_scale
        fro = torch.linalg.matrix_norm(J).expand(x_iv.lo.shape[0]).contiguous()
        return fro


def _pca_U(system, data: RolloutData) -> torch.Tensor:
    Xn = (data.x0 - system.equilibrium()) / system.lqr_like_scale()
    return torch.linalg.svd(Xn).Vh.T.contiguous()          # columns = axes


def _random_U(D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64, generator=g))
    return Q


def _build_region(transport, system, D: int) -> Region:
    xr = (2.0 * torch.rand((2048, D), dtype=torch.float64,
                           generator=torch.Generator().manual_seed(2)) - 1.0) \
        * system.lqr_like_scale()
    with torch.no_grad():
        y = transport(xr)
    eta_scale = y[:, :D_ETA].abs().quantile(0.9, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, D_ETA:].norm(dim=-1).quantile(0.9).clamp_min(1e-3))
    return Region(d_eta=D_ETA, eta_scale=eta_scale, rho_radius=rho_radius,
                  full_scale=torch.cat([eta_scale,
                                        torch.full((D - D_ETA,),
                                                   rho_radius / (D - D_ETA) ** 0.5,
                                                   dtype=torch.float64)]))


def _era_masks(unit: torch.Tensor, split: float):
    lo = torch.zeros_like(unit, dtype=torch.bool)
    hi = torch.zeros_like(unit, dtype=torch.bool)
    for u in unit.unique():
        idx = torch.nonzero(unit == u).squeeze(-1)
        n = idx.numel()
        cut = int(n * split)
        lo[idx[:cut]] = True
        hi[idx[cut:]] = True
    return lo, hi


def load_dataset(name: str):
    """(base system, drift system, n_rows) per domain -- exp4/5/6 conventions."""
    if name == "cmapss":
        data_real = load_cmapss_fd001()
        unit = data_real["unit"]
        lo_mask, _ = _era_masks(unit, 0.5)
        sys_base = SensorSystem.fit(data_real["X"][lo_mask], unit[lo_mask])
        _, hi_mask = _era_masks(unit, 0.5)
        sys_drift = SensorSystem.fit(data_real["X"][hi_mask], unit[hi_mask])
        return sys_base, sys_drift, int(data_real["X"].shape[0])
    if name == "aqi":
        data_real = load_prsa_aq()
        sys_base = SensorSystem.fit(data_real["X"][data_real["era_lo"] == 1.0])
        sys_drift = SensorSystem.fit(data_real["X"][data_real["era_hi"] == 1.0])
        return sys_base, sys_drift, int(data_real["X"].shape[0])
    if name == "ett":
        data_real = load_ettm2()
        seg = data_real["unit"]
        sys_base = SensorSystem.fit(data_real["X"][data_real["era_hi"]], seg[data_real["era_hi"]])
        sys_drift = SensorSystem.fit(data_real["X"][data_real["era_lo"]], seg[data_real["era_lo"]])
        return sys_base, sys_drift, int(data_real["X"].shape[0])
    raise ValueError(name)


def run_variant(kind: str, system, D: int, wm_steps: int, cert_steps: int,
                n_traj: int, node_budget: int, time_budget: float,
                DT: float = 1.0) -> dict:
    """One coordinate map through the full downstream protocol."""
    t0 = time.time()
    x0 = (2.0 * torch.rand((n_traj, D), dtype=torch.float64,
                           generator=torch.Generator().manual_seed(0)) - 1.0) \
        * system.lqr_like_scale()
    x1 = euler_maruyama(system, x0, 1, DT,
                        generator=torch.Generator().manual_seed(1))
    data = RolloutData(x0.detach(), x1.detach(), DT, system=system)

    if kind == "pca":
        transport = LinearTransport(_pca_U(system, data), system.equilibrium(),
                                    system.lqr_like_scale())
    elif kind == "random":
        transport = LinearTransport(_random_U(D, 11), system.equilibrium(),
                                    system.lqr_like_scale())
    else:
        raise ValueError(kind)

    F = LatentDynamics(D, width=64, depth=2, seed=0, d_eta=D_ETA)
    train_world_model(transport, F, data, D_ETA,
                      TrainConfig(steps=wm_steps, seed=0, w_contract=0.3,
                                  d_eta=D_ETA, rho_spec_cap=1.0,
                                  refit_steps=wm_steps // 2))
    region = _build_region(transport, system, D)

    V = LyapunovNet(D_ETA, use_residual=False, p_scale=1.0, seed=0)
    cert_hist = train_certificate(V, F, transport, system, D_ETA,
                                  CertConfig(steps=cert_steps, alpha=ALPHA,
                                             kappa=KAPPA, seed=0,
                                             v_res_cap=1.0, v_coef_cap=0.05))

    from sbounds.generator import noise_floor
    beta = noise_floor(V, F, transport, system, KAPPA, D_ETA)
    trained = {"transport": transport, "F": F, "V": V, "region": region,
               "system": system, "D": D, "d_eta": D_ETA, "beta": beta}
    cert = certify_both(trained, node_budget, time_budget, alpha=ALPHA, tol=TOL)

    # pointwise certificate health under the same plant, same probes as exp4/5/6
    lo, hi = region.init_box("full")
    g = torch.Generator().manual_seed(7)
    u = torch.rand((512, D), dtype=torch.float64, generator=g)
    y_probe = lo + (hi - lo) * u
    pr = true_violation_probe(V, F, transport, system, KAPPA, ALPHA, D_ETA,
                              y_probe, beta=beta, tol=TOL)
    return {"cert": cert, "beta": beta,
            "viol_frac": pr["viol_frac"],
            "cert_viol_frac": cert_hist["viol_frac"][-1],
            "seconds": time.time() - t0}


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", os.cpu_count() or 4)))
    wm_steps, cert_steps, n_traj = (300, 300, 600) if quick else (1200, 900, 2400)
    node_budget, time_budget = (1200, 200.0) if quick else (4030, 900.0)

    out_path = os.path.join(RESULTS, "exp7_baselines.json")
    store = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            store = json.load(fh)
    ckpt = torch.load(CKPT, weights_only=False) if os.path.exists(CKPT) else {}

    for ds in ("cmapss", "aqi", "ett"):
        sys_base, sys_drift, n_rows = load_dataset(ds)
        D = sys_base.dim
        ident = {"dA_norm": float((sys_base.A - sys_drift.A).norm()),
                 "attractor_shift": float((sys_base.equilibrium()
                                           - sys_drift.equilibrium()).norm()),
                 "n_rows": n_rows, "D": D}
        store.setdefault(ds, {"identification": ident})
        for kind in ("pca", "random"):
            key = f"{ds}:{kind}"
            if key in store.get(ds, {}) and not quick:
                continue
            ck = ckpt.get(key)
            if ck is None or quick:
                print(f"[{key}] training + certifying ({node_budget} nodes)...",
                      flush=True)
                ck = run_variant(kind, sys_base, D, wm_steps, cert_steps,
                                 n_traj, node_budget, time_budget)
                if not quick:
                    ckpt[key] = {"cert": ck["cert"], "beta": ck["beta"]}
                    torch.save(ckpt, CKPT)
            # learned-column import for the self-contained table
            src = os.path.join(RESULTS, LEARNED_SRC[ds])
            learned = None
            if os.path.exists(src):
                with open(src, encoding="utf-8") as fh:
                    lj = json.load(fh)
                learned = {"cert": lj.get("cert"), "beta": lj.get("beta"),
                           "cert_viol_frac": lj.get("cert_viol_frac")}
            store[ds][kind] = ck
            store[ds]["learned"] = learned
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(store, fh, indent=2, default=_jsonable)
            f_pct = ck["cert"]["factor"]["certified_fraction"]
            print(f"[{key}] factor={f_pct:.3f} full="
                  f"{ck['cert']['full']['certified_fraction']:.3f} "
                  f"viol={ck['viol_frac']:.3f} beta={ck['beta']:.2e} "
                  f"({ck['seconds']:.0f}s)", flush=True)
    print("saved results/exp7_baselines.json", flush=True)


if __name__ == "__main__":
    main()
