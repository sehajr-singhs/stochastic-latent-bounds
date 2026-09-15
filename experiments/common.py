"""Shared setup for the experiments: build, train, and freeze one certificate.

The region is not chosen by hand. It is the 90th percentile of the transported
coordinates over a held-out sample of the plant's own trajectories, so the
claim is about the part of the state space the system actually visits, and the
region definition itself is reproducible from the data.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from sbounds.models import LatentDynamics, LyapunovNet
from sbounds.region import Region
from sbounds.systems import ChainArm
from sbounds.train import (CertConfig, TrainConfig, generate_pairs, score_world_model,
                           train_certificate, train_world_model)
from sbounds.transport import InvertibleTransport

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@dataclass
class SetupConfig:
    n_links: int = 8
    d_eta: int = 4
    sigma: float = 0.05
    noise: str = "additive"
    kappa: float = 1.0
    alpha: float = 0.2
    n_traj: int = 4000
    wm_steps: int = 1200
    cert_steps: int = 1000
    transport_layers: int = 4
    transport_width: int = 32
    f_width: int = 64
    seed: int = 0
    quantile: float = 0.9
    verbose: bool = False


@dataclass
class Setup:
    cfg: SetupConfig
    system: ChainArm
    transport: InvertibleTransport
    F: LatentDynamics
    V: LyapunovNet
    region: Region
    metrics: dict = field(default_factory=dict)

    @property
    def d_eta(self) -> int:
        return self.cfg.d_eta

    @property
    def alpha(self) -> float:
        return self.cfg.alpha

    @property
    def kappa(self) -> float:
        return self.cfg.kappa


def build(cfg: SetupConfig = SetupConfig()) -> Setup:
    t0 = time.time()
    system = ChainArm(n_links=cfg.n_links, sigma=cfg.sigma, noise=cfg.noise)
    D = system.dim
    data = generate_pairs(system, n_traj=cfg.n_traj, dt=0.01, region_scale=1.0,
                          seed=cfg.seed)
    transport = InvertibleTransport(
        dim=D, d_latent=cfg.d_eta, n_layers=cfg.transport_layers,
        width=cfg.transport_width, hidden_depth=2,
        x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=cfg.seed)
    F = LatentDynamics(D, width=cfg.f_width, depth=2, seed=cfg.seed)
    wm_hist = train_world_model(transport, F, data, cfg.d_eta,
                                TrainConfig(steps=cfg.wm_steps, seed=cfg.seed),
                                verbose=cfg.verbose)
    t_wm = time.time() - t0

    # held-out scoring against the exact Ito push-forward of the real plant
    xh = generate_pairs(system, n_traj=512, dt=0.01, seed=cfg.seed + 99).x0
    score = score_world_model(system, transport, F, xh)

    # region from the 90th percentile of transported coordinates
    xr = generate_pairs(system, n_traj=2048, dt=0.01, seed=cfg.seed + 123).x0
    with torch.no_grad():
        y = transport(xr)
    q = cfg.quantile
    eta_scale = y[:, :cfg.d_eta].abs().quantile(q, dim=0).clamp_min(1e-3)
    rho_radius = float(y[:, cfg.d_eta:].norm(dim=-1).quantile(q).clamp_min(1e-3))
    region = Region(d_eta=cfg.d_eta, eta_scale=eta_scale, rho_radius=rho_radius,
                    full_scale=torch.cat([eta_scale, torch.full((D - cfg.d_eta,),
                                                                rho_radius / (D - cfg.d_eta) ** 0.5,
                                                                dtype=torch.float64)]))

    t1 = time.time()
    V = LyapunovNet(cfg.d_eta, width=32, depth=2, n_res=16, seed=cfg.seed)
    cert_hist = train_certificate(V, F, transport, system, cfg.d_eta,
                                  CertConfig(steps=cfg.cert_steps, alpha=cfg.alpha,
                                             kappa=cfg.kappa, seed=cfg.seed),
                                  verbose=cfg.verbose)
    t_cert = time.time() - t1

    metrics = {
        "dim_state": D,
        "d_eta": cfg.d_eta,
        "sigma": cfg.sigma,
        "noise": cfg.noise,
        "alpha": cfg.alpha,
        "kappa": cfg.kappa,
        "eta_scale": eta_scale.tolist(),
        "rho_radius": rho_radius,
        "world_model": score,
        "final_lam_rho": wm_hist["final_lam_rho"],
        "wm_fit_final": wm_hist["fit"][-1],
        "cert_viol_frac_final": cert_hist["viol_frac"][-1],
        "cert_max_gen_final": cert_hist["max_gen"][-1],
        "seconds_world_model": t_wm,
        "seconds_certificate": t_cert,
        "params": {
            "transport": sum(p.numel() for p in transport.parameters()),
            "F": sum(p.numel() for p in F.parameters()),
            "V": sum(p.numel() for p in V.parameters()),
        },
    }
    return Setup(cfg, system, transport, F, V, region, metrics)


def save(name: str, payload: dict, extra_config: dict | None = None) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    if extra_config:
        payload = {"config": extra_config, **payload}
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)
    return path


def _jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.detach().tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)
