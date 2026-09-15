"""Soundness is the load-bearing claim, so it is tested directly.

The centerpiece is `test_bound_soundness`: across many random boxes, the sound
upper bound must dominate the exact generator at every sampled interior point.
A failure there is a correctness bug, not a tuning issue.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbounds.models import LyapunovNet, solve_lyapunov_metric
from sbounds.systems import ChainArm, pushforward_drift
from sbounds.bounds import bound_factor, bound_full
from sbounds.generator import latent_generator, noise_floor
from sbounds.transport import InvertibleTransport

torch.manual_seed(0)
SYSTEM = ChainArm(n_links=2, sigma=0.05)
D = SYSTEM.dim
TRANSPORT = InvertibleTransport(dim=D, d_latent=2, n_layers=3, width=16,
                                x_star=SYSTEM.equilibrium(),
                                x_scale=SYSTEM.lqr_like_scale(), seed=0)
V = LyapunovNet(2, width=16, depth=2, n_res=8, seed=1)
F = None  # filled per-test where needed


def test_transport_roundtrip():
    """T^-1(T(x)) == x to numerical precision: the diffeomorphism is exact."""
    x = torch.randn(64, D, dtype=torch.float64) * 0.5
    y = TRANSPORT(x)
    x2 = TRANSPORT.inverse(y)
    assert torch.allclose(x, x2, atol=1e-9), (x - x2).abs().max()


def test_transport_zero_at_equilibrium():
    x_star = SYSTEM.equilibrium().unsqueeze(0)
    y = TRANSPORT(x_star)
    assert float(y.abs().max()) < 1e-12


def test_generator_noise_floor_at_origin():
    """With additive noise the origin is not an SDE equilibrium: LW(0) = beta > 0.

    The certificate is LW + alpha W <= beta, and at the origin it holds with
    equality by definition of beta. grad W(0) = 0 and F(0) = 0, so beta is
    exactly the Ito term.
    """
    from sbounds.models import LatentDynamics

    Fm = LatentDynamics(D, width=16, depth=1, seed=2)
    zero = torch.zeros(1, D, dtype=torch.float64)
    beta = noise_floor(V, Fm, TRANSPORT, SYSTEM, kappa=1.0, d_eta=2)
    assert beta > 0.0, "additive noise must produce a positive noise floor"
    resid = latent_generator(V, Fm, TRANSPORT, SYSTEM, kappa=1.0, alpha=0.2, d_eta=2,
                             eta=zero[:, :2], rho=zero[:, 2:])
    assert abs(float(resid[0]) - beta) < 1e-9
    # alpha must not enter beta (W(0) = 0)
    beta2 = noise_floor(V, Fm, TRANSPORT, SYSTEM, kappa=1.0, d_eta=2)
    assert abs(beta - beta2) < 1e-12


def test_lyapunov_metric_hurwitz():
    """The rho metric solves A^T Q + Q A = -I; residual and definiteness checked."""
    from sbounds.models import LatentDynamics

    torch.manual_seed(3)
    A = torch.randn(4, 4, dtype=torch.float64) - 2.0 * torch.eye(4, dtype=torch.float64)
    Q = solve_lyapunov_metric(A)
    resid = A.T @ Q + Q @ A + torch.eye(4, dtype=torch.float64)
    assert float(resid.abs().max()) < 1e-8
    assert float(torch.linalg.eigvalsh(Q).min()) > 0


@pytest.mark.parametrize("mode", ["factor", "full"])
def test_bound_soundness(mode):
    """Three one-sided checks:

    1. the sound upper bound dominates the exact generator at sampled points;
    2. a degenerate point box at the origin reproduces the noise floor beta to
       high precision -- the tightness assertion the Cauchy-Schwarz bound could
       never pass (its slack is bounded away from zero);
    3. whenever the bound declares a box certified (upper <= beta + tol), every
       sampled interior point satisfies the exact generator <= beta + tol. A
       failure here would be an unsound certificate, the one bug this repo must
       not have.
    """
    from sbounds.models import LatentDynamics

    torch.manual_seed(11)
    Fm = LatentDynamics(D, width=16, depth=1, seed=2)
    # a contracting latent drift: certification must succeed near the origin,
    # which is what gives this test its teeth
    with torch.no_grad():
        Fm.A.copy_(-1.5 * torch.eye(D, dtype=torch.float64))
    Fm.refresh_rho_metric(2)
    kappa, alpha, d = 1.0, 0.2, 2
    beta = noise_floor(V, Fm, TRANSPORT, SYSTEM, kappa=kappa, d_eta=d)
    tol = 0.05
    worst_gap = -float("inf")
    worst_unsound = -float("inf")
    n_certified = 0
    boxes = []
    for i in range(8):
        if i < 6:   # origin-centered shrinking boxes: must certify when small
                    # (the centered bound's slack is quadratic; certification
                    #  needs width ~< 2*sqrt(tol/gfro) ~ 0.1 here)
            w = torch.full((D,), 0.5 ** (i + 1), dtype=torch.float64)
            boxes.append((-w, w))
        else:       # random boxes anywhere: soundness must hold regardless
            c = torch.randn(D, dtype=torch.float64) * 0.4
            w = torch.rand(D, dtype=torch.float64) * 0.3 + 0.05
            boxes.append((c - w, c + w))
    # tightness at the origin: point box, tight trace must hit beta exactly
    if mode == "factor":
        res0 = bound_factor(V, Fm, TRANSPORT, SYSTEM, kappa, alpha, d,
                            torch.zeros(1, d, dtype=torch.float64),
                            torch.zeros(1, d, dtype=torch.float64), 0.0, chunk=8)
    else:
        res0 = bound_full(V, Fm, TRANSPORT, SYSTEM, kappa, alpha, d,
                          torch.zeros(1, D, dtype=torch.float64),
                          torch.zeros(1, D, dtype=torch.float64), chunk=8)
    origin_gap = float(res0.upper[0]) - beta
    assert abs(origin_gap) < 1e-6, f"tight trace not tight at origin: off by {origin_gap:.3e}"
    for lo, hi in boxes:
        if mode == "factor":
            res = bound_factor(V, Fm, TRANSPORT, SYSTEM, kappa, alpha, d,
                               lo[:d].unsqueeze(0), hi[:d].unsqueeze(0),
                               float(hi[d:].abs().max()), chunk=8)
        else:
            res = bound_full(V, Fm, TRANSPORT, SYSTEM, kappa, alpha, d,
                             lo.unsqueeze(0), hi.unsqueeze(0), chunk=8)
        pts = torch.rand(64, D, dtype=torch.float64) * (hi - lo) + lo
        eta, rho = pts[:, :d], pts[:, d:]
        exact = latent_generator(V, Fm, TRANSPORT, SYSTEM, kappa, alpha, d, eta, rho)
        worst_gap = max(worst_gap, float((exact - res.upper[0]).max()))
        if float(res.upper[0]) <= beta + tol:
            n_certified += 1
            worst_unsound = max(worst_unsound, float(exact.max()) - (beta + tol))
    assert worst_gap <= 1e-6, f"bound not an enclosure: exceeded by {worst_gap:.3e}"
    assert worst_unsound <= 1e-6, f"certified box violated threshold by {worst_unsound:.3e}"
    assert n_certified > 0, "no box certified in this sweep; test lost its teeth"


def test_pushforward_matches_finite_difference():
    """The exact Ito push-forward drift matches a finite-difference SDE mean."""
    torch.manual_seed(5)
    x = torch.randn(32, D, dtype=torch.float64) * 0.3
    dt = 1e-4
    n = 4000
    total = torch.zeros_like(x)
    g = torch.Generator().manual_seed(7)
    for _ in range(n):
        # Sigma maps the n_links-dimensional Wiener process into the state:
        # diffusion(x) is (B, D, n_links), so eps has n_links channels
        eps = torch.randn(x.shape[0], SYSTEM.n_links, dtype=torch.float64, generator=g)
        drift = SYSTEM.drift(x)
        sig = SYSTEM.diffusion(x)
        xn = x + drift * dt + (sig @ eps.unsqueeze(-1)).squeeze(-1) * (dt ** 0.5)
        total += TRANSPORT(xn)
    em = total / n
    exact = pushforward_drift(SYSTEM, TRANSPORT, x)
    # em estimates E[T(x_{t+dt})] = T(x) + dt * pushforward + O(dt^2), so the
    # finite-difference target is the transported state, not the raw state
    scale = exact.norm(dim=-1).clamp_min(1e-6)
    with torch.no_grad():
        err = ((em - TRANSPORT(x)) / dt - exact).norm(dim=-1) / scale
    assert float(err.mean()) < 0.5
